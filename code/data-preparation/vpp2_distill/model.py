"""Student-only training and inference; the 14B teacher is never imported here."""
from __future__ import annotations

import torch
from torch import nn
from PIL import Image

from .action import FrozenPolicyCore
from .contracts import MemoryConfig, cpu_noise, file_digest
from .losses import policy_losses
from .memory import StructuredMemory


class QwenPrefix(nn.Module):
    def __init__(self, model, processor, revision=None):
        super().__init__()
        self.model = model
        self.processor = processor
        self.revision = revision

    @classmethod
    def pretrained(cls, model_id, revision=None):
        from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration
        config = AutoConfig.from_pretrained(model_id, revision=revision)
        resolved = getattr(config, "_commit_hash", None) or revision
        model, info = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, revision=resolved, config=config, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa", ignore_mismatched_sizes=False, output_loading_info=True)
        if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError(f"Qwen checkpoint failed strict loading audit: {info}")
        processor = AutoProcessor.from_pretrained(model_id, revision=resolved)
        processor.tokenizer.padding_side = "left"
        model.config.use_cache = False
        return cls(model, processor, resolved)

    @classmethod
    def from_bundle(cls, directory, revision=None):
        from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration
        config = AutoConfig.from_pretrained(directory / "qwen_config", local_files_only=True)
        config._attn_implementation = "sdpa"
        config.use_cache = False
        model = Qwen3VLForConditionalGeneration(config)
        processor = AutoProcessor.from_pretrained(directory / "processor", local_files_only=True)
        processor.tokenizer.padding_side = "left"
        return cls(model, processor, revision)

    def forward(self, examples, device):
        messages = []
        for example in examples:
            history = example["history"]
            if tuple(history.shape) != (9, 3, 240, 416) or history.dtype != torch.uint8:
                raise ValueError("Qwen requires nine native uint8 history frames")
            images = [Image.fromarray(x.permute(1, 2, 0).cpu().numpy()) for x in history]
            content = [{"type": "image", "image": im} for im in images]
            content.append({"type": "text", "text": example["prompt"]})
            messages.append([{"role": "user", "content": content}])
        inputs = self.processor.apply_chat_template(messages, tokenize=True, padding=True,
                    add_generation_prompt=True, return_dict=True, return_tensors="pt").to(device)
        # The memory decoder attends the whole prefix, including instruction tokens.
        output = self.model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True, logits_to_keep=1)
        return output.hidden_states[-1], inputs["attention_mask"].bool()


class DistillationPolicy(nn.Module):
    def __init__(self, encoder, memory, action_core):
        super().__init__()
        self.encoder = encoder
        self.memory = memory
        self.action_core = action_core
        self.lambda_kd = 0.5
        self.lambda_kv = 0.1

    @classmethod
    def build(cls, config, *, teacher_key_norms=None):
        if file_digest(config["action_checkpoint"]) != config["identity"]["action_sha256"]:
            raise ValueError("Action checkpoint differs from teacher target identity")
        core = FrozenPolicyCore.from_checkpoint(config["action_checkpoint"])
        norms = teacher_key_norms if teacher_key_norms is not None else torch.load(
            config["teacher_key_norms"], weights_only=True, map_location="cpu")
        if norms["action_sha256"] != config["identity"]["action_sha256"] or norms["video_sha256"] != config["identity"]["video_sha256"]:
            raise ValueError("Key norms belong to a different teacher checkpoint pair")
        cfg = MemoryConfig(**config.get("memory", {}))
        if (cfg.video_layers, cfg.kv_heads, cfg.head_dim, cfg.context_dim, cfg.latent_channels, tuple(cfg.grid), cfg.patch) != (40, 40, 128, 4096, 16, (5,15,26), 2):
            raise ValueError("Student output must preserve the native RoboDojo memory interface")
        encoder = QwenPrefix.pretrained(config["qwen_model"], config.get("qwen_revision"))
        if encoder.model.config.text_config.hidden_size != cfg.qwen_dim:
            raise ValueError("Qwen hidden width disagrees with the memory configuration")
        encoder.float()  # FP32 trainable weights and AdamW state; BF16 autocast.
        memory = StructuredMemory(cfg, norms["weights"], checkpoint=True)
        result = cls(encoder, memory, core)
        result.set_stage("warmup")
        return result

    @property
    def device(self):
        return next(self.memory.parameters()).device

    @property
    def torch_dtype(self):
        return next(self.action_core.parameters()).dtype

    def train(self, mode=True):
        super().train(mode)
        self.action_core.proprio_encoder.eval()
        self.action_core.runner.expert.eval()
        if not any(p.requires_grad for p in self.encoder.parameters()):
            self.encoder.eval()
        return self

    def set_stage(self, stage):
        if stage not in {"warmup", "main", "refresh"}:
            raise ValueError(f"Unknown training stage: {stage}")
        self.encoder.requires_grad_(stage != "warmup")
        # The vocabulary head has no role in feature distillation. Keep an
        # untied head out of AdamW/DDP; tied input embeddings remain trainable.
        backbone = self.encoder.model
        if hasattr(backbone, "get_output_embeddings"):
            head = backbone.get_output_embeddings()
            inputs = {id(p) for p in backbone.get_input_embeddings().parameters()}
            if head is not None:
                for parameter in head.parameters():
                    if id(parameter) not in inputs:
                        parameter.requires_grad_(False)
        self.action_core.requires_grad_(False)
        self.memory.key_norms.requires_grad_(False)
        self.lambda_kv = 1.0 if stage == "warmup" else 0.1
        self.train(self.training)

    def make_memory(self, examples):
        device, dtype = self.device, self.torch_dtype
        max_len = max(e["context"].shape[0] for e in examples)
        context = torch.zeros((len(examples), max_len, self.memory.cfg.context_dim), device=device, dtype=dtype)
        mask = torch.zeros((len(examples), max_len), device=device, dtype=torch.bool)
        for i, e in enumerate(examples):
            length = e["context"].shape[0]
            context[i, :length] = e["context"].to(device=device, dtype=dtype)
            mask[i, :length] = e["context_mask"].to(device=device, dtype=torch.bool)
        state = torch.stack([e["state"] for e in examples]).to(device=device, dtype=dtype)
        context, mask = self.action_core.append_proprio(context, mask, state)
        hidden, hidden_mask = self.encoder(examples, device)
        noise = torch.cat([cpu_noise((1, *self.memory.cfg.noise_shape), e["video_seed"], device=device, dtype=dtype)
                           for e in examples], dim=0)
        memories = self.memory(hidden, hidden_mask, context, mask, noise)
        return memories, context, mask

    def forward(self, examples):
        if not examples:
            raise ValueError("Empty training batch")
        memories, context, mask = self.make_memory(examples)
        device, dtype = self.device, self.torch_dtype
        actions = torch.stack([e["action"] for e in examples]).to(device=device, dtype=dtype)
        padding = torch.stack([e["action_is_pad"] for e in examples]).to(device)
        scheduler = self.action_core.scheduler
        times = scheduler.sample_training_t(len(examples), device, dtype)
        noise = torch.randn_like(actions)
        x = scheduler.add_noise(actions, noise, times)
        gt_target = scheduler.training_target(actions, noise, times)
        teacher = torch.zeros_like(gt_target)
        teacher_valid = torch.zeros(len(examples), device=device, dtype=torch.bool)
        gt_valid = torch.tensor([e.get("gt_valid", True) and e.get("sample_valid", True) for e in examples], device=device)
        kv_targets = []
        for i, e in enumerate(examples):
            query = e.get("query")
            if query is not None:
                x[i] = query["x"].to(device=device, dtype=dtype)
                times[i] = query["t"].to(device=device, dtype=dtype)
                gt_target[i] = query["gt"].to(device=device, dtype=dtype)
                teacher[i] = query["teacher"].to(device=device, dtype=dtype)
                teacher_valid[i] = True
            kv_targets.append(e.get("kv_target"))
        prediction = self.action_core(x, times, context, mask, memories)
        defaults = (len(examples) / max(int(gt_valid.sum()), 1), len(examples) / max(int(teacher_valid.sum()), 1))
        scales = [(e.get("gt_scale", defaults[0]), e.get("teacher_scale", defaults[1])) for e in examples]
        if len(set(scales)) != 1:
            raise ValueError("All microbatch examples must belong to one logical loss normalization")
        return policy_losses(prediction, gt_target, padding, times, scheduler, gt_valid=gt_valid,
                             teacher_velocity=teacher, teacher_valid=teacher_valid, memories=memories,
                             kv_targets=kv_targets, lambda_kd=self.lambda_kd, lambda_kv=self.lambda_kv,
                             gt_scale=scales[0][0], teacher_scale=scales[0][1])

    @torch.no_grad()
    def predict_chunk(self, examples, *, trace=False):
        memories, context, mask = self.make_memory(examples)
        initial = torch.cat([cpu_noise((1, 32, 16), e["action_seed"], device=self.device, dtype=self.torch_dtype)
                             for e in examples], dim=0)
        return self.action_core.sample(initial, context, mask, memories, trace=trace)

    @torch.no_grad()
    def predict_action(self, examples, **kwargs):
        return {"normalized_actions": self.predict_chunk(examples).float().cpu().numpy()}

