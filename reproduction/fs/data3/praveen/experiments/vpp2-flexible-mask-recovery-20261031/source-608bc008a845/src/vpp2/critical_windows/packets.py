"""CPU-only whole-episode review assets. Creation never marks a review complete."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import html
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from vpp2.full_mask.panel import digest, file_hash, immutable
from .contracts import read_json, make_manifest, checked_identity, annotation_template


def write_sheet(frames, indices, path, title):
    cols = 4; width = 416; cell_height = 262
    rows = (len(indices) + cols - 1) // cols
    canvas = Image.new('RGB', (cols * width, rows * cell_height + 30), '#f7f8fa')
    draw = ImageDraw.Draw(canvas); draw.text((8, 7), title, fill='#16212b')
    for i, t in enumerate(indices):
        image = Image.fromarray(np.asarray(frames[t]).transpose(1, 2, 0))
        x = (i % cols) * width; y = (i // cols) * cell_height + 30
        canvas.paste(image, (x, y + 20))
        draw.text((x + 5, y + 3), f'Original frame {t}', fill='#111111')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.partial'); canvas.save(tmp, format='JPEG', quality=92); tmp.replace(path)
    return dict(file=path.name, frames=list(map(int, indices)), sha256=file_hash(path))


def episode_packet(args):
    config, manifest_identity, e = args
    out = Path(config['root']) / 'packets' / f"{e['episode']:05d}"
    if (out / 'packet.json').exists():
        existing = checked_identity(read_json(out / 'packet.json'))
        if existing['manifest_identity'] != manifest_identity:
            raise ValueError('Stale packet')
        for page in existing['coarse_pages']:
            if file_hash(out / page['file']) != page['sha256']:
                raise ValueError('Changed review image')
        return existing
    source = Path(config['demonstrations']) / e['path']
    for name in ('frames.npy', 'actions.npy', 'states.npy'):
        if file_hash(source / name) != e['files'][name]:
            raise ValueError(f'Source changed for episode {e["episode"]}: {name}')
    frames = np.load(source / 'frames.npy', mmap_mode='r', allow_pickle=False)
    if frames.shape != (e['length'], 3, 240, 416) or frames.dtype != np.uint8:
        raise ValueError('Unexpected native frame format')
    indices = sorted(set(range(0, e['length'], config['coarse_frame_stride'])) | {e['length'] - 1})
    pages = []
    for i in range(0, len(indices), 24):
        pages.append(write_sheet(frames, indices[i:i + 24], out / f'coarse_{i // 24:03d}.jpg',
                                 f'{e["task"]} | episode {e["episode"]} | {e["role"]} | index units, not seconds'))
    packet = dict(schema='critical-visual-packet-v1', manifest_identity=manifest_identity,
                  episode=e['episode'], task=e['task'], role=e['role'], length=e['length'],
                  source_files=e['files'], coarse_pages=pages,
                  frame_unit='original_decoded_frame_index', source_fps=None,
                  fps_note='No FPS inferred from playback or filenames. Annotation uses exact frame indices.',
                  review_complete=False)
    packet['identity'] = digest(packet); immutable(out / 'packet.json', packet)
    return packet


def gallery(config, manifest, packets):
    root = Path(config['root']); rows = []
    from .contracts import load_annotations
    verified, _ = load_annotations(config, manifest)
    by_id = {p['episode']: p for p in packets}
    templates = root / 'annotation_templates_v2'; templates.mkdir(exist_ok=True)
    for e in manifest['episodes']:
        p = by_id[e['episode']]; template = annotation_template(manifest, e, p['identity'])
        immutable(templates / f"{e['episode']:05d}.json", template)
        current = root / 'annotations' / f"{e['episode']:05d}.json"
        if current.exists():
            from .contracts import validate_annotation
            saved=read_json(current);validate_annotation(manifest,saved,p)
            if saved.get('status') == 'verified' and e['episode'] not in verified:
                raise ValueError('Review page cannot present unvalidated verified labels')
            template=saved
        rows.append(dict(episode=e['episode'], task=e['task'], role=e['role'], suite=e['suite'],
                         instruction=e['prompt'], length=e['length'], template=template,
                         pages=[f"packets/{e['episode']:05d}/{x['file']}" for x in p['coarse_pages']]))
    payload = json.dumps(rows, separators=(',', ':')).replace('<', '\\u003c')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Critical interval episode audit</title>
<style>body{font:16px system-ui;max-width:1700px;margin:24px;color:#142333;background:#f7f8fa}select,button{font:inherit;padding:8px;margin:4px}img{max-width:100%;display:block;margin:16px 0;border:1px solid #c8d1db}article{padding:20px;background:white;border:1px solid #c8d1db}textarea{width:98%;height:280px;font:13px monospace}.notice{background:#fff1cf;padding:14px}#state{font-weight:600}</style>
<h1>Whole-episode critical interval review</h1><p class="notice">All labels begin unreviewed. Scan every page, then inspect dense frames around boundaries before verifying an interval. Frame numbers refer to the ORIGINAL decoded episode. These review images include future observations; they never become policy inputs. No policy predictions or simulator scores are shown. Sealed audit episodes are excluded.</p>
<p>Keep final approaches, precision alignment, purposeful contact, controlled release, bimanual coordination, memory-dependent decisions and required subgoals. Include valid recoveries; do not infer contact solely from gripper motion. Memory decisions need the cue interval and a separate check of native-history observability.</p>
<p><select id="task"><option value="">All tasks</option></select><select id="role"><option value="">Both roles</option><option>fit</option><option>monitor</option></select><button id="prev">Previous episode</button><button id="next">Next episode</button><span id="state"></span></p>
<article><h2 id="title"></h2><p id="instruction"></p><div id="images"></div><details><summary>Annotation JSON — save locally, then validate through the importer</summary><p>Set integer [start,end) frame ranges. Include kind, label, id, evidence_frames and observed evidence for each interval. Recording a draft does not certify it. Mark verified only after actual visual inspection.</p><textarea id="annotation"></textarea><button id="save">Download annotation</button><p id="error"></p></details></article>
<script>const DATA=__DATA__;let offset=0;const el=id=>document.getElementById(id);for(const task of [...new Set(DATA.map(x=>x.task))].sort()){let o=document.createElement('option');o.value=task;o.textContent=task;el('task').append(o)}
function selected(){return DATA.filter(x=>(!el('task').value||x.task===el('task').value)&&(!el('role').value||x.role===el('role').value))}
function render(){let rows=selected();offset=Math.max(0,Math.min(offset,rows.length-1));let r=rows[offset];el('state').textContent=`${offset+1} / ${rows.length}; ${r.template.status} (imported-review snapshot)`;el('title').textContent=`${r.task} — episode ${r.episode} (${r.role}, ${r.length} frames)`;el('instruction').textContent=r.instruction;el('images').replaceChildren();for(const src of r.pages){let img=document.createElement('img');img.src=src;img.loading='lazy';img.alt=`Original frame contact sheet for episode ${r.episode}`;el('images').append(img)}el('annotation').value=JSON.stringify(r.template,null,2);el('error').textContent=''}
el('prev').onclick=()=>{offset--;render()};el('next').onclick=()=>{offset++;render()};for(const name of ['task','role'])el(name).onchange=()=>{offset=0;render()};el('save').onclick=()=>{try{const a=JSON.parse(el('annotation').value);if(a.episode!==selected()[offset].episode)throw Error('Wrong episode');const u=URL.createObjectURL(new Blob([JSON.stringify(a,null,2)],{type:'application/json'}));let link=document.createElement('a');link.href=u;link.download=String(a.episode).padStart(5,'0')+'.json';link.click();URL.revokeObjectURL(u)}catch(e){el('error').textContent=String(e)}};render();</script></html>'''
    (root / 'review.html').write_text(page.replace('__DATA__', payload))


def dense(config, manifest, episode, start, end, stride=2):
    e = next((x for x in manifest['episodes'] if x['episode'] == episode), None)
    if e is None:
        raise ValueError('Episode is not in the permitted development panel')
    if not 0 <= start < end <= e['length'] or stride < 1:
        raise ValueError('Invalid dense-frame request')
    source_path=Path(config['demonstrations']) / e['path'] / 'frames.npy'
    if file_hash(source_path) != e['files']['frames.npy']:
        raise ValueError('Episode frames changed before dense review')
    frames = np.load(source_path, mmap_mode='r')
    out = Path(config['root']) / 'packets' / f'{episode:05d}'
    indices = sorted(set(range(start, end, stride)) | {end - 1})
    pages = []
    for i in range(0, len(indices), 24):
        path = out / f'dense_{start:05d}_{end:05d}_{stride}_{i // 24:03d}.jpg'
        pages.append(write_sheet(frames, indices[i:i + 24], path, f'{e["task"]}: dense boundary review, episode {episode}'))
    receipt = dict(episode=episode, manifest_identity=manifest['identity'], start=start, end=end,
                   stride=stride, pages=pages, review_complete=False)
    immutable(out / f'dense_{start:05d}_{end:05d}_{stride}.json', receipt)
    return receipt


def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True)
    p.add_argument('--workers', type=int, default=8); p.add_argument('--episode', type=int)
    p.add_argument('--start', type=int); p.add_argument('--end', type=int); p.add_argument('--stride', type=int, default=2)
    a = p.parse_args(); c = read_json(a.config); root = Path(c['root']); root.mkdir(parents=True, exist_ok=True)
    manifest = make_manifest(c)
    if a.episode is not None:
        print(json.dumps(dense(c, manifest, a.episode, a.start, a.end, a.stride))); return
    packets = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        jobs = [pool.submit(episode_packet, (c, manifest['identity'], e)) for e in manifest['episodes']]
        for job in as_completed(jobs):
            packets.append(job.result())
            print(json.dumps(dict(event='episode_packet_ready', complete=len(packets), total=len(jobs))), flush=True)
    gallery(c, manifest, packets)
    immutable(root / 'preparation_receipt.json', dict(manifest_identity=manifest['identity'],
              episodes=len(packets), coarse_pages=sum(len(p['coarse_pages']) for p in packets),
              cpu_only=True, annotations_verified=0, visual_review_complete=False,
              review=str(root / 'review.html'), sealed_audit_opened=False))


if __name__ == '__main__':
    main()
