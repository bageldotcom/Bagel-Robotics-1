"""One compact-model recovery run; no mask objectives or automatic extension."""
import argparse,json
from pathlib import Path
from .config import load,identity
from .export import read_export_manifest
from .train import run


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--export');a=p.parse_args();c=load(a.config)
    path=a.export
    if path is None and c.get('recovery_subset'):
        path=c['recovery_subset']['export_path']
    if path is None:
        receipt=json.loads((Path(c['root'])/'export_verified.json').read_text())
        if receipt['identity']!=identity(c) or not receipt['passed']:raise ValueError('No verified compact export')
        path=receipt['path']
    m=read_export_manifest(path,require_verified=True)
    from .recovery_subset import require_export
    require_export(c,path)
    runtime,status=run(c,stage='recovery',recovery_export=path)
    if status['status']=='recovery_complete':
        from .recovery_export import publish
        publish(runtime,path,status['update'])

if __name__=='__main__':main()
