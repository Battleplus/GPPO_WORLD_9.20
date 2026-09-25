"""Explicit authorization for one frozen runtime-necessity diagnostic."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
ATTEMPT='independent-task-benefit-20260925-once'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def authorize(path):
    if not path or not Path(path).is_file():raise PermissionError('New diagnostic authorization required; stage2 balances are not authority')
    a=read(path);r=read(ROOT/'BUDGET_REQUEST.json');m=read(ROOT/'execution-manifest.json')
    if (a.get('approved') is not True or a.get('attempt')!=ATTEMPT or a.get('caps')!=r['caps']
        or a.get('stage_caps')!=r['stage_caps'] or a.get('manifest_sha256')!=sha(ROOT/'execution-manifest.json')
        or a.get('budget_sha256')!=sha(ROOT/'BUDGET_REQUEST.json')
        or not a.get('user_approval_text') or not a.get('user_approval_reference')):
        raise PermissionError('Explicit bound one-time diagnostic approval missing')
    for row in m['files']:
        if sha(row['path'])!=row['sha256']:raise RuntimeError('Frozen identity changed: '+row['path'])
    return a,m,r

def limits(request):
    keys=['environment_steps','environment_resets','policy_encode','actor_readout','world_candidate_batch',
          'rule_decisions','model_loads','optimizer_updates','world_updates','offline_updates']
    result={}
    for stage,cap in request['stage_caps'].items():
        for key in keys:result[stage+'/'+key]=cap.get(key,0)
    for key in keys:
        if sum(v for k,v in result.items() if k.endswith('/'+key))!=request['caps'][key]:
            raise ValueError('Substage resource allocation mismatch: '+key)
    return result
