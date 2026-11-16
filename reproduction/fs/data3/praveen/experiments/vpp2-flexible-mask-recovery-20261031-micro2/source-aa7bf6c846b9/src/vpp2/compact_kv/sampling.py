"""Global sample/noise identity independent of microbatch, rank and resume."""
import random
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .spec import stable_seed


class PositionDataset(Dataset):
    def __init__(self, native, seed):
        self.native, self.seed = native, seed
        self.epoch, self.order = None, None

    def __len__(self):
        return len(self.native)

    def __getitem__(self, position):
        epoch, offset = divmod(int(position), len(self.native))
        if epoch != self.epoch:
            gen = torch.Generator().manual_seed(self.seed + epoch)
            self.order = torch.randperm(len(self.native), generator=gen).tolist()
            self.epoch = epoch
        self.native.sampling_epoch = epoch
        index = self.order[offset]
        numpy_state, python_state = np.random.get_state(), random.getstate()
        try:
            aug_seed = stable_seed(self.seed, "augmentation", position)
            np.random.seed(aug_seed % 2**32)
            random.seed(aug_seed)
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(aug_seed)
                value = self.native._get(index)  # Never substitute another window on decode failure.
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)
        value["pilot_position"] = torch.tensor(position, dtype=torch.long)
        return value


class GlobalBatchSampler(Sampler):
    def __init__(self, start, end, rank, world, micro=1, global_batch=288):
        if global_batch % (world * micro):
            raise ValueError("Global batch must divide exactly across ranks/microbatches")
        self.start, self.end, self.rank, self.world = start, end, rank, world
        self.micro, self.global_batch = micro, global_batch
        self.accumulation = global_batch // (world * micro)

    def __iter__(self):
        for update in range(self.start, self.end):
            for microstep in range(self.accumulation):
                start = update * self.global_batch + microstep * self.world * self.micro + self.rank * self.micro
                yield list(range(start, start + self.micro))

    def __len__(self):
        return (self.end - self.start) * self.accumulation
