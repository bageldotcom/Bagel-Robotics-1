import argparse
import json
from pathlib import Path
from .spec import atomic_json, file_sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    config=json.loads(Path(a.config).read_text())
    source=Path(config['teacher_bundle'])
    value={k:file_sha(source/k) for k in ('video.pt','action.pt','dataset_stats.json')}
    value['source']=str(source)
    atomic_json(Path(config['root'])/'teacher_identity.json',value)
    print(json.dumps(value),flush=True)

if __name__=='__main__': main()
