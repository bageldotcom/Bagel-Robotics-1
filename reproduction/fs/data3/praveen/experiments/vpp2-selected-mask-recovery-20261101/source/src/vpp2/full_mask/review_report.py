"""Local visual review with native history, action traces and explicit unknowns."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
import html
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from .curation import load_panel
from .panel import immutable


def storyboard(source,row,path):
    if path.exists():
        try:
            with Image.open(path) as previous:previous.verify()
            return
        except (OSError,SyntaxError):pass
    frames=np.load(source/row['episode_path']/'frames.npy',mmap_mode='r')
    actions=np.load(source/row['episode_path']/'actions.npy',mmap_mode='r')
    # All nine native inputs, followed by three review-only future frames.
    ids=row['history_indices']+[min(row['length']-1,row['frame']+d) for d in (8,16,31)]
    canvas=Image.new('RGB',(3*208,4*140+100),'#fafafa');draw=ImageDraw.Draw(canvas)
    for i,t in enumerate(ids):
        arr=np.asarray(frames[t]).transpose(1,2,0)
        im=Image.fromarray(arr).resize((208,120))
        x=(i%3)*208;y=(i//3)*140
        canvas.paste(im,(x,y+18))
        label=('INPUT ' if i<9 else 'REVIEW ONLY ')+str(t)
        draw.text((x+3,y+2),label,fill='#111111' if i<9 else '#9b4000')
    # State the scale; each arm's positional displacement is relative to its start.
    a=np.array(actions[row['frame']:row['frame']+32],dtype=np.float64)
    dp=np.stack([np.linalg.norm(a[:,s:s+3]-a[:1,s:s+3],axis=-1) for s in (0,8)])
    maximum=max(float(dp.max()),1e-6)
    draw.text((4,565),f'Hand displacement from chunk start; range 0–{maximum*1000:.2f} mm',fill='#111111')
    for arm,color in [(0,'#2277bb'),(1,'#bb5522')]:
        pts=[(15+i*18,645-float(v)/maximum*55) for i,v in enumerate(dp[arm])]
        draw.line(pts,fill=color,width=2)
    temporary=path.with_suffix('.partial')
    canvas.save(temporary,format='JPEG',quality=84)
    temporary.replace(path)


def build(config,panel,max_images=None,workers=4):
    out=Path(config['curation']['output']);assets=out/'review_assets';assets.mkdir(exist_ok=True)
    source=Path(config['demonstrations']);rows=[];pending=[]
    # Include all selected and monitoring windows. Browsers lazily decode assets.
    for i,r in enumerate(panel['windows']):
        filename=f"{r['episode']:05d}_{r['frame']:05d}.jpg"
        if max_images is None or i<max_images:pending.append((r,assets/filename))
        rows.append({**{k:r[k] for k in ('id','window_id','task','suite','episode','frame','role','prompt','proxy_tags','phase_status','memory_cue_observable','selection_reason')},
                     'image':'review_assets/'+filename,'history':r['history_indices']})
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs=[pool.submit(storyboard,source,r,path) for r,path in pending]
        for i,job in enumerate(as_completed(jobs),1):
            job.result()
            if i%500==0:print(json.dumps(dict(event='storyboards',done=i,total=len(pending))),flush=True)
    near=[json.loads(line) for line in (out/'near_duplicate_review.jsonl').read_text().splitlines()]
    manifest=json.dumps(dict(identity=panel['identity'],rows=rows,coverage=panel['coverage'],near_pairs=near),separators=(',',':')).replace('<','\\u003c')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><title>RoboDojo calibration review</title>
<style>body{font:16px system-ui;margin:24px;max-width:1500px;color:#172330}h1{font-size:26px}select,input,button{font:inherit;margin:6px;padding:7px}article{border:1px solid #bbc8d5;border-radius:9px;padding:12px;margin:14px 0;display:grid;grid-template-columns:minmax(300px,625px) 1fr;gap:18px}img{width:100%;height:auto}.note{background:#fff3db;padding:14px}.small{font-size:13px;color:#435266}table{border-collapse:collapse}td,th{padding:6px 12px;border-bottom:1px solid #ddd}.pending{color:#8d3900}textarea{width:90%;height:70px}summary{cursor:pointer}@media(max-width:850px){article{display:block}}</style>
<h1>Full RoboDojo calibration review</h1>
<p class="note">Curated candidates. Frozen visual and motion descriptors guide diversity; they do not verify manipulation meaning. Contact, phase and memory-cue labels are <b>unverified</b>. Similarity suggestions have not automatically deleted windows. Future frames in each storyboard are review-only and never policy inputs.</p>
<p id="identity" class="small"></p><details><summary>Task coverage</summary><div id="coverage"></div></details>
<p><select id="task"><option value="">All tasks</option></select><select id="role"><option value="">All roles</option><option>selection_initial</option><option>monitor</option></select><input id="search" placeholder="Episode or proxy tag"><button id="prev">Previous</button><button id="next">Next</button><button id="download">Download review decisions</button><span id="count"></span></p>
<details><summary>Possible near duplicates: compare both histories and actions</summary><div id="pairs"></div></details><button id="clearPair">Show all windows</button>
<div id="items"></div><script>const DATA=__DATA__;const storageKey='robodojo-review-'+DATA.identity;let decisions={};try{decisions=JSON.parse(localStorage.getItem(storageKey)||'{}')}catch(e){}let offset=0,pair=null;const PAGE=12;
const el=id=>document.getElementById(id);el('identity').textContent='Panel identity: '+DATA.identity;
for(const t of Object.keys(DATA.coverage).sort()){let o=document.createElement('option');o.value=t;o.textContent=t;el('task').append(o)}
let table=document.createElement('table');let tr=document.createElement('tr');for(const x of ['Task','Suite','Selected','Episodes','Monitor','Phase verified']){let td=document.createElement('th');td.textContent=x;tr.append(td)}table.append(tr);
for(const [t,c] of Object.entries(DATA.coverage)){let tr=document.createElement('tr');for(const x of [t,c.suite,c.selected_windows,c.selected_episodes,c.monitor_windows,String(c.phase_verified)]){let td=document.createElement('td');td.textContent=x;tr.append(td)}table.append(tr)}el('coverage').append(table);
for(const p of DATA.near_pairs){let b=document.createElement('button');b.textContent=`${p.task}: ${p.first.join('/')} versus ${p.second.join('/')}`;b.onclick=()=>{pair=new Set([p.first.join('/'),p.second.join('/')]);el('task').value=p.task;el('role').value='selection_initial';el('search').value='';offset=0;show()};el('pairs').append(b)}el('clearPair').onclick=()=>{pair=null;show()};
function show(){let rs=DATA.rows.filter(r=>(!pair||pair.has([r.episode,r.frame].join('/')))&&(!el('task').value||r.task===el('task').value)&&(!el('role').value||r.role===el('role').value)&&(!el('search').value||JSON.stringify(r).includes(el('search').value)));offset=Math.max(0,Math.min(offset,Math.max(0,rs.length-1)));el('count').textContent=` ${offset+1}–${Math.min(offset+PAGE,rs.length)} / ${rs.length}`;el('items').replaceChildren();
for(const r of rs.slice(offset,offset+PAGE)){let card=document.createElement('article'),img=document.createElement('img');img.src=r.image;img.loading='lazy';img.alt=`Native history and action segment for ${r.task}, episode ${r.episode}, frame ${r.frame}`;card.append(img);let info=document.createElement('div');let title=document.createElement('h2');title.textContent=`${r.task}: episode ${r.episode}, frame ${r.frame}`;info.append(title);for(const text of [r.role,'Instruction: '+r.prompt,'Selection: '+r.selection_reason,'Proxy tags: '+r.proxy_tags.join(', '),'History: '+r.history.join(', '),'Memory cue observable: '+String(r.memory_cue_observable)]){let p=document.createElement('p');p.textContent=text;info.append(p)}let select=document.createElement('select');for(const x of ['unreviewed','retain_verified','retain_uncertain','replace_missing_phase','suggest_duplicate']){let o=document.createElement('option');o.value=x;o.textContent=x;select.append(o)}let notes=document.createElement('textarea');notes.placeholder='Evidence: phase, cue frame/identity, reason this sample is needed. No implied contact labels.';let saved=decisions[r.window_id];if(saved){select.value=saved.decision;notes.value=saved.evidence}let save=()=>{decisions[r.window_id]={window_id:r.window_id,episode:r.episode,frame:r.frame,decision:select.value,evidence:notes.value};try{localStorage.setItem(storageKey,JSON.stringify(decisions))}catch(e){}};select.onchange=save;notes.oninput=save;info.append(select,notes);card.append(info);el('items').append(card)}}
for(const id of ['task','role','search'])el(id).oninput=()=>{offset=0;pair=null;show()};el('prev').onclick=()=>{offset-=PAGE;show()};el('next').onclick=()=>{offset+=PAGE;show()};el('download').onclick=()=>{let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify({schema:'curation-review-v2',panel_identity:DATA.identity,decisions:Object.values(decisions)},null,2)],{type:'application/json'}));a.download='curation-review-decisions.json';a.click();URL.revokeObjectURL(a.href)};show();</script></html>'''
    (out/'review.html').write_text(page.replace('__DATA__',manifest))
    return dict(panel_identity=panel['identity'],windows=len(rows),storyboards_built=sum((assets/Path(r['image']).name).exists() for r in rows),
                review_complete=False,report=str(out/'review.html'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--max-images',type=int);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    c=json.loads(Path(a.config).read_text());r=build(c,load_panel(c['panel_file']),a.max_images,a.workers)
    (Path(c['curation']['output'])/'review_build.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))


if __name__=='__main__':main()
