"""Publish and CPU-validate the staged subset, without launching mask training."""
import argparse
from collections import Counter, defaultdict
import copy
import html
import json
from pathlib import Path

from vpp2.full_mask.panel import immutable
from vpp2.full_mask.verify_data import verify
from .contracts import checked_identity, read_json, load_annotations
from .reselect import build
from .staged_review import prepare, load_records, write_status


def runtime_config(config, output):
    c=copy.deepcopy(config)
    c['panel_file']=str(output/'panel.json')
    c['curation_artifacts']=c['base_data']
    c['curation']['output']=str(output)
    c['readiness_dir']=str(output)
    for key,name in [('teacher_threshold_file','teacher_thresholds.json'),
                     ('cpu_receipt_file','cpu_checks.json'),('gpu_receipt_file','gpu_checks.json')]:
        c[key]=str(output/name)
    return c


def report(config, panel, output, cpu):
    root=Path(config['root']);stage=prepare(config)
    manifest=checked_identity(read_json(root/'audit_manifest.json'))
    annotations,_=load_annotations(config,manifest)
    records=load_records(config,manifest,stage)
    status=write_status(config,panel)
    if not status['passed']:raise RuntimeError('Visual evidence no longer matches the revised panel')
    publication=read_json(output/'receipt.json')['publication_checks']
    suite=defaultdict(Counter)
    for r in panel['windows']:suite[r['suite']][r['role']]+=1
    rows=[]
    for ep,record in sorted(records.items(),key=lambda x:(x[1]['task'],x[0])):
        packet=read_json(root/'packets'/f'{ep:05d}'/'packet.json')
        rows.append(dict(episode=ep,task=record['task'],observations=record['observations'],
                         anchors=record['manipulation_anchor_frames'],memory=record['memory_cue_status'],
                         pages=[str(root/'packets'/f'{ep:05d}'/p['file']) for p in packet['coarse_pages']],
                         record=str(root/'staged_review/records'/f'{ep:05d}.json')))
    receipt=dict(panel_identity=panel['identity'],panel=str(output/'panel.json'),
                 config=str(output/'config.json'),source=str(Path(__file__).resolve().parents[3]),
                 coarse_screened_fitting_episodes=len(records),dense_annotated_development_episodes=len(annotations),
                 reviewed_task_groups=len({r['task'] for r in records.values()}),
                 suite_windows=dict(suite),publication_checks=publication,
                 cpu_checks_passed=cpu['passed'],native_gpu_checks='pending_for_this_panel',
                 full_280_annotation_required_before_training=False,training_started=False,
                 limitations=['Coarse screens certify only the inspected timeline images and listed frames.',
                              'Memory cue observability remains uncertain where not densely verified.',
                              'Coverage and finite-gradient checks cannot establish closed-loop policy retention.',
                              'The independent final-audit episodes remain sealed.'])
    immutable(output/'preparation_receipt.json',receipt)
    table=''.join(f'<tr><td>{html.escape(k)}</td><td>{v["selection_initial"]:,}</td><td>{v["monitor"]:,}</td></tr>' for k,v in sorted(suite.items()))
    data=json.dumps(rows).replace('</','<\\/')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>RoboDojo staged calibration review</title><style>
body{font:17px/1.5 system-ui,sans-serif;max-width:1200px;margin:32px auto;padding:0 20px;color:#152235;background:#f6f8fb}h1,h2{line-height:1.2}section{background:white;border:1px solid #d8e1ec;border-radius:12px;padding:20px;margin:18px 0}.note{background:#fff5de}table{border-collapse:collapse}td,th{padding:8px 18px;border-bottom:1px solid #ddd;text-align:left}button,select{font:inherit;padding:7px;max-width:100%}img{display:block;width:100%;height:auto;margin:16px 0}code,.path{overflow-wrap:anywhere;font-size:14px}a{color:#1259a7}.muted{color:#536577}
</style><h1>RoboDojo calibration subset: staged review</h1>
<section><p><strong id="screenCount">SCREEN_COUNT fitting episodes screened across TASK_COUNT task groups.</strong> This is a coarse timeline screen. Dense critical-interval annotations exist for DENSE_COUNT development episodes; the other episodes are not labelled as fully verified.</p>
<p>The initial 37 screens were followed by four targeted screens for tower construction, conveyor picking, packing, and stacking-toy phases. Every recorded task-relevant frame is covered by the revised selection.</p>
<p><strong>30,000 fitting windows · 1,119 monitoring windows · 2,906 fitting episodes.</strong> Action targets do not overlap within an episode. Native observation histories can overlap.</p>
<table><thead><tr><th>Suite/group</th><th>Fitting windows</th><th>Monitoring windows</th></tr></thead><tbody>SUITE_TABLE</tbody></table></section>
<section class="note"><strong>CPU integrity checks passed. Fresh GPU checks are still required.</strong><p>This page is not authorization to bypass native equivalence or action-sensitivity checks. The exhaustive 280-episode annotation study is separate from this staged preparation; the independent final-audit episodes remain sealed. Coarse images do not establish memory-cue visibility or simulation success.</p></section>
<section><h2>Review the inspected images</h2><p class="muted">Episode and frame numbers are original dataset indices, not seconds. Notes describe only what was visible.</p><select id="episode"></select> <button id="prev">Previous</button> <button id="next">Next</button><h3 id="title"></h3><p id="notes"></p><p id="anchors"></p><p id="memory"></p><a id="record">Open hashed review record</a><div id="images"></div></section>
<section><h2>Artifacts</h2><p>Panel: <span class="path">PANEL_PATH</span></p><p>Configuration: <span class="path">CONFIG_PATH</span></p><p>Preparation receipt: <span class="path">RECEIPT_PATH</span></p><p class="path">Panel identity: PANEL_ID</p></section>
<script>const rows=ROW_DATA;const choice=document.getElementById('episode');for(let i=0;i<rows.length;i++){const option=document.createElement('option');option.value=i;option.textContent=rows[i].task+' — episode '+rows[i].episode;choice.appendChild(option)}
function draw(){const r=rows[Number(choice.value)];document.getElementById('title').textContent=r.task+' — episode '+r.episode;document.getElementById('notes').textContent=r.observations;document.getElementById('anchors').textContent='Protected observed frames: '+r.anchors.join(', ');document.getElementById('memory').textContent='Memory-cue status: '+r.memory;document.getElementById('record').href='file://'+r.record;const images=document.getElementById('images');images.replaceChildren();for(let j=0;j<r.pages.length;j++){const image=document.createElement('img');image.src='file://'+r.pages[j];image.alt=r.task+' episode '+r.episode+' coarse timeline page '+(j+1);images.appendChild(image)}}choice.onchange=draw;document.getElementById('prev').onclick=()=>{choice.value=(Number(choice.value)+rows.length-1)%rows.length;draw()};document.getElementById('next').onclick=()=>{choice.value=(Number(choice.value)+1)%rows.length;draw()};draw();</script></html>'''
    replacements={'SCREEN_COUNT':str(len(records)),'TASK_COUNT':str(len(stage['tasks'])),
                  'DENSE_COUNT':str(len(annotations)),'SUITE_TABLE':table,'ROW_DATA':data,
                  'PANEL_PATH':html.escape(str(output/'panel.json')),'CONFIG_PATH':html.escape(str(output/'config.json')),
                  'RECEIPT_PATH':html.escape(str(output/'preparation_receipt.json')),'PANEL_ID':panel['identity']}
    for key,value in replacements.items():page=page.replace(key,value)
    (output/'review.html').write_text(page)
    return receipt


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    config=read_json(a.config)
    if config.get('review_protocol')!='staged_visual_v1':raise ValueError('Staged protocol required')
    output=build(config);c=runtime_config(config,output);immutable(output/'config.json',c)
    panel=checked_identity(read_json(output/'panel.json'))
    print(json.dumps(dict(event='staged_panel_published',path=str(output/'panel.json'))),flush=True)
    cpu=verify(c,panel,check_reader=True);immutable(output/'cpu_checks.json',cpu)
    receipt=report(c,panel,output,cpu)
    pointer=Path(config['root'])/'staged_review/current_preparation.json'
    tmp=pointer.with_suffix('.partial');tmp.write_text(json.dumps(receipt,indent=2)+'\n');tmp.replace(pointer)
    print(json.dumps(dict(event='staged_preparation_complete',**receipt)),flush=True)


if __name__=='__main__':main()
