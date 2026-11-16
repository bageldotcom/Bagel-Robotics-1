"""Recompute task/suite/overall tables from the archived direct-action records."""
from pathlib import Path
import csv
import hashlib
import html
import json
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]

def read(p):return json.loads(Path(p).read_text())

def summarize(record, phase, source):
    tasks = sorted({r["task"] for r in record["rows"]})
    if len(tasks) != 35 or len(record["rows"]) != 140:
        raise ValueError("The historical comparison requires 35 tasks and 140 windows")
    if len({r["window_id"] for r in record["rows"]}) != 140:
        raise ValueError("Duplicate validation windows")
    result=[]
    for task in tasks:
        rows=[r for r in record["rows"] if r["task"]==task]
        if len(rows)!=4:raise ValueError("Unbalanced validation task")
        suites={r["suite"] for r in rows}
        if len(suites)!=1:raise ValueError("Ambiguous suite")
        result.append(dict(phase=phase,update=record["update"],suite=next(iter(suites)),task=task,
            student_gt_mse=mean(r["student_error"] for r in rows),
            teacher_gt_mse=mean(r["teacher_error"] for r in rows),
            student_teacher_mse=(mean(r["student_teacher_error"] for r in rows)
                if all("student_teacher_error" in r for r in rows) else None),
            windows=len(rows),source=source))
    return result

def average(rows,group):
    by={}
    for r in rows:by.setdefault((r["phase"],r["update"],r[group] if group else "overall"),[]).append(r)
    out=[]
    for (phase,update,name),items in sorted(by.items()):
        row=dict(phase=phase,update=update,group=name,tasks=len(items))
        for k in ("student_gt_mse","teacher_gt_mse","student_teacher_mse"):
            row[k]=mean(r[k] for r in items) if all(r[k] is not None for r in items) else None
        out.append(row)
    return out

def csvwrite(path,rows):
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    target=ROOT/"results"; evidence=target/"evidence";rows=[];seen=set();source_records=[]
    panel_ids={}
    for name,phase in [("mask_fast","mask_learning"),("mask_final","mask_learning"),
                       ("recovery_pilot","recovery"),("recovery_1000","recovery")]:
        for p in sorted((evidence/name).glob("native_*.json")):
            d=read(p);key=(phase,d["update"])
            if key in seen:continue
            seen.add(key)
            panel=sorted((r["window_id"],r["episode"],r["frame"],r["task"]) for r in d["rows"])
            if phase in panel_ids and panel_ids[phase]!=panel:raise ValueError("Validation panel changed")
            panel_ids[phase]=panel
            rows.extend(summarize(d,phase,str(p.relative_to(ROOT))))
            source_records.append(dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    if panel_ids["mask_learning"]!=panel_ids["recovery"]:raise ValueError("Mask/recovery panels differ")
    overall=average(rows,None);suites=average(rows,"suite")
    csvwrite(target/"task_metrics.csv",rows);csvwrite(target/"suite_metrics.csv",suites);csvwrite(target/"overall_metrics.csv",overall)
    (target/"metrics.json").write_text(json.dumps(dict(overall=overall,suites=suites,tasks=rows),indent=2)+"\n")
    (target/"provenance.json").write_text(json.dumps(dict(
        sources=source_records,metric="MSE of normalized generated actions over first min(24, valid_steps) of 32 predicted actions",
        validation="140 fixed development-monitoring windows, 4 per task, 35 tasks",
        comparison="Within-panel checkpoint comparison; no closed-loop success claims",
        simulation=False,missing_teacher_disagreement="Not recorded at some earlier mask checkpoints; never imputed"),indent=2)+"\n")
    lines=["RECORDED DIRECT-ACTION VALIDATION (not flow-matching loss; not simulation)",
           "phase  update  student/GT  student/teacher  teacher/GT"]
    for r in overall:
        st="not recorded" if r["student_teacher_mse"] is None else f'{r["student_teacher_mse"]:.10g}'
        lines.append(f'{r["phase"]:14} {r["update"]:5d} {r["student_gt_mse"]:.10g} {st} {r["teacher_gt_mse"]:.10g}')
    (target/"summary.txt").write_text("\n".join(lines)+"\n")
    body=[]
    for phase in ("mask_learning","recovery"):
        values=[r for r in overall if r["phase"]==phase]
        body.append("<h2>"+phase.replace("_"," ").title()+"</h2><table><tr><th>Update</th><th>Student vs GT</th><th>Student vs teacher</th><th>Teacher vs GT</th></tr>")
        for r in values:
            body.append("<tr><td>"+str(r["update"])+"</td>"+"".join("<td>"+("Not recorded" if r[k] is None else f'{r[k]:.8f}')+"</td>" for k in ["student_gt_mse","student_teacher_mse","teacher_gt_mse"])+"</tr>")
        body.append("</table><img src='"+phase+".svg' alt='"+phase+" validation trajectory'>")
    body.append("<h2>Task-level comparison</h2><p>Select two recovery checkpoints. Values are direct action MSE; lower is better.</p><select id='left'></select> versus <select id='right'></select><div id='tasks'></div>")
    script="""const data=DATA;
const steps=[...new Set(data.filter(x=>x.phase==='recovery').map(x=>x.update))].sort((a,b)=>a-b);
for(const id of ['left','right'])for(const s of steps){let o=document.createElement('option');o.value=s;o.textContent=s;document.getElementById(id).append(o);}
document.getElementById('left').value=55;document.getElementById('right').value=1000;
function draw(){let a=Number(document.getElementById('left').value),b=Number(document.getElementById('right').value);
let aa=data.filter(x=>x.phase==='recovery'&&x.update===a),bb=data.filter(x=>x.phase==='recovery'&&x.update===b);
let table=document.createElement('table');let h=document.createElement('tr');
for(const text of ['Suite','Task','Student/GT at '+a,'Student/GT at '+b,'Change','Student/teacher at '+b]){let t=document.createElement('th');t.textContent=text;h.append(t);}table.append(h);
for(const x of aa){let y=bb.find(v=>v.task===x.task);if(!y)continue;let tr=document.createElement('tr');for(const v of [x.suite,x.task,x.student_gt_mse.toFixed(8),y.student_gt_mse.toFixed(8),(100*(y.student_gt_mse/x.student_gt_mse-1)).toFixed(1)+'%',y.student_teacher_mse===null?'Not recorded':y.student_teacher_mse.toFixed(8)]){let td=document.createElement('td');td.textContent=v;tr.append(td);}table.append(tr);}
document.getElementById('tasks').replaceChildren(table);}
for(const id of ['left','right'])document.getElementById(id).onchange=draw;draw();""".replace("DATA",json.dumps(rows).replace("</","<\\/"))
    page="<!doctype html><meta charset='utf-8'><title>VPP2 mask learning and recovery</title><style>body{font:16px system-ui;max-width:1150px;margin:40px auto;padding:0 20px;color:#17202a}table{border-collapse:collapse;width:100%;margin:15px 0}th,td{text-align:right;padding:8px;border-bottom:1px solid #ddd}th:first-child,td:first-child{text-align:left}img{width:100%}select{font-size:18px;padding:8px}aside{background:#eef4fb;padding:16px}</style><h1>VPP2 mask learning and recovery: recorded results</h1><aside>These are existing experiment records, not results from the new package. The 140-window panel was used repeatedly for development. It does not measure simulation success or provide an independent final test.</aside><p>Selected mask: update 1,500. Compact video parameters: 12,468,409,664. Recovery best: 1,000; runner-up: 950.</p>"+"".join(body)+"<script>"+script+"</script>"
    (target/"index.html").write_text(page)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for phase in ("mask_learning","recovery"):
            values=[r for r in overall if r["phase"]==phase]
            fig,ax=plt.subplots(figsize=(10,4.8))
            for key,label in [("student_gt_mse","Student vs ground truth"),("student_teacher_mse","Student vs teacher"),("teacher_gt_mse","Teacher vs ground truth")]:
                points=[r for r in values if r[key] is not None]
                ax.plot([r["update"] for r in points],[r[key] for r in points],marker="o",label=label)
            if phase=="mask_learning":ax.set_yscale("log")
            ax.set(xlabel="Optimizer update",ylabel="Direct normalized-action MSE",title=phase.replace("_"," ").title()+" — fixed development panel")
            ax.grid(alpha=.25);ax.legend();fig.tight_layout()
            fig.savefig(target/(phase+".svg"));fig.savefig(target/(phase+".png"),dpi=160);plt.close(fig)
    except ImportError:
        print("Tables generated. Install matplotlib to regenerate the supplied figures.")
    print("Wrote task/suite/overall tables, provenance, and HTML report.")
if __name__=="__main__":main()
