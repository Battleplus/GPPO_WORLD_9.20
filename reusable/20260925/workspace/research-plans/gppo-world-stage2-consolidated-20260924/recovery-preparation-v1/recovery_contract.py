"""Authorization and cumulative accounting; stdlib, no runtime work on import."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def authorize(path):
    if not path or not Path(path).is_file():raise PermissionError('Recovery execution requires a new explicit recovery authorization')
    a=read(path);b=read(ROOT/'recovery-manifest.json');r=read(ROOT/'RECOVERY_BUDGET_REQUEST.json')
    if (a.get('approved') is not True or a.get('recovery_manifest_sha256')!=sha(ROOT/'recovery-manifest.json')
        or a.get('budget_request_sha256')!=sha(ROOT/'RECOVERY_BUDGET_REQUEST.json')
        or a.get('caps')!=r['caps'] or not a.get('user_approval_reference') or not a.get('user_approval_text')):
        raise PermissionError('Approval must bind recovery, including one additional model load')
    for row in b['files']:
        if sha(row['path'])!=row['sha256']:raise RuntimeError('Recovery-bound input changed: '+row['path'])
    return a,b,r

def reject_duplicate_or_wrong_order(completed, remaining, matrix):
    k=lambda r:(r['parent'],r['repeat'],r['arm'])
    c=[k(r) for r in completed];r=[k(x) for x in remaining];p=[k(x) for x in matrix]
    if len(c)!=len(set(c)) or len(r)!=len(set(r)) or set(c)&set(r) or c+r!=p:
        raise RuntimeError('Recovery must be the exact untouched suffix of the frozen matrix')

def cumulative_snapshot(old_stages, segment):
    result={}
    for name,old in old_stages.items():
        v=segment['stages'].get(name, {})
        result[name]={s:old[s]+v.get(s,0) for s in ['reserved','verified','unknown']}
        result[name]['pending']=result[name]['reserved']-result[name]['verified']-result[name]['unknown']
        result[name]['original_limit']=old['limit']
        limit=old['limit']+(1 if name=='C/model_loads' else 0)
        result[name]['authorized_limit_including_explicit_extension']=limit
        if result[name]['reserved']>limit:raise RuntimeError('Cumulative original-stage limit exceeded')
    return result
