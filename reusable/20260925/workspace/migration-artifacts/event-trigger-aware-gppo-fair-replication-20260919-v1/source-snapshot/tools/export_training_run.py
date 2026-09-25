"""Export a complete or interrupted run without omitting checkpoints or logs."""
import argparse,hashlib,json,zipfile
from pathlib import Path


def completion_state(run):
    """A result filename alone is not proof that the frozen matrix finished."""
    if (run/'failure.json').exists():return False
    if not (run/'results.json').is_file() or not (run/'run-manifest.json').is_file():return False
    results=json.loads((run/'results.json').read_text(encoding='utf-8'))
    manifest=json.loads((run/'run-manifest.json').read_text(encoding='utf-8'))
    protocol=manifest.get('protocol',{})
    expected={(g,s) for g in protocol.get('groups',[]) for s in protocol.get('seeds',[])}
    actual=[(r['group'],r['seed']) for r in results.get('results',[])]
    return bool(expected and len(actual)==len(expected) and set(actual)==expected
        and results.get('status')=='development_matrix_completed'
        and results.get('updates')==protocol.get('training',{}).get('unique_updates_total')
        and all((run/f'{g}-{s}'/'terminal.pt').is_file() for g,s in expected))


def export(run,destination):
    run=run.resolve();destination=destination.resolve()
    if destination.is_relative_to(run):raise ValueError('Export must be outside the run directory')
    files=sorted(p for p in run.rglob('*') if p.is_file())
    if not files:raise ValueError('Empty run')
    members=[]
    for p in files:
        if p.is_symlink():raise ValueError('Symlinks not permitted in export')
        members.append({'path':p.relative_to(run).as_posix(),'bytes':p.stat().st_size,
                        'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    manifest={'format':'gppo-training-export/1.0.0','source_directory':str(run),
        'completed':completion_state(run),'contains_failure':(run/'failure.json').is_file(),
        'checkpoint_count':sum(p.suffix=='.pt' for p in files),'files':members}
    with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for p,member in zip(files,members):archive.write(p,member['path'])
        archive.writestr('EXPORT-MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(destination) as archive:
        for member in members:
            value=archive.read(member['path'])
            if len(value)!=member['bytes'] or hashlib.sha256(value).hexdigest()!=member['sha256']:
                raise ValueError('Archive readback mismatch')
    result={'path':str(destination),'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
            'bytes':destination.stat().st_size,'verified_members':len(members),'checkpoint_count':manifest['checkpoint_count'],
            'completed':manifest['completed']}
    destination.with_suffix(destination.suffix+'.sha256.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(export(args.run,args.output)))
