from pathlib import Path
import numpy as np
from vpp2.adaptive_mask.report import plot_monitor
from vpp2.adaptive_mask.monitor import summarize

def test_visual_reports_include_layers_gates_quality_and_future(tmp_path):
    rows=[dict(task='task',suite='memory',episode=i,frame=0,window_id=i,teacher_error=1.,student_error=1.001,
        base_reliability=.8,reliability=.5,student_veto=False,zero_veto=False,component_veto=False,kv=np.ones((40,2)).tolist()) for i in range(4)]
    q=summarize(rows);monitor=dict(rows=rows,video_rows=rows,action_quality=q,video_quality=q,examples=[dict(
        task='task',label='Single-step latent magnitude',teacher_future=np.ones((1,2,3,4)).tolist(),student_future=np.ones((1,2,3,4)).tolist())])
    native=dict(rows=rows,examples=[dict(task='task',teacher=np.zeros((32,16)).tolist(),student=np.zeros((32,16)).tolist(),target=np.zeros((32,16)).tolist())])
    selection=dict(ffn_indices=[list(range(8))]*40,cross_head_indices=[list(range(2))]*40)
    paths=plot_monitor(monitor,native,[],selection,70,tmp_path)
    assert len(paths)==6 and all(p.stat().st_size>1000 for p in paths)
