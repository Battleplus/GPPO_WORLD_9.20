"""No runtime authority without a new approval including all three amendments."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def authorize(path):
    if not path or not Path(path).is_file():raise PermissionError('New local recovery2 approval required')
    a=read(path);b=read(ROOT/'recovery-manifest.json');r=read(ROOT/'RECOVERY_BUDGET_REQUEST.json')
    if r.get('preparation_ready') is not True:raise PermissionError('Preparation not frozen for approval')
    if (a.get('approved') is not True or a.get('attempt')!='consolidated-v1-recovery-2-local-once'
        or a.get('recovery_manifest_sha256')!=sha(ROOT/'recovery-manifest.json')
        or a.get('budget_request_sha256')!=sha(ROOT/'RECOVERY_BUDGET_REQUEST.json')
        or a.get('caps')!=r['caps'] or not a.get('user_approval_reference') or not a.get('user_approval_text')
        or a.get('accepted_amendments')!=r['required_authorization_amendments']):
        raise PermissionError('Approval must explicitly bind unresolved I/O mitigation, reset 9601 and model load 6')
    for row in b['files']:
        if sha(row['path'])!=row['sha256']:raise RuntimeError('Bound input changed: '+row['path'])
    validate_resource_envelope(r,read(ROOT/'historical-budget-readonly.json'))
    return a,b,r

def validate_resource_envelope(request,history):
    caps=request['caps'];old=request['historical_resources'];limits=request['original_C_limits']
    segment={'stages':{'C/'+k:{'reserved':caps[k],'verified':0,'unknown':0} for k in caps if 'C/'+k in history}}
    cumulative_snapshot(history,segment)
    for old_key,new_key,limit_key,hold_key in [
        ('C_wall_seconds','wall_seconds','wall_seconds','wall_seconds'),
        ('C_cpu_seconds','total_process_cpu_seconds','total_process_cpu_seconds','cpu_seconds')]:
        if old[old_key]+caps[new_key]+request['historical_shutdown_holdback'][hold_key]>limits[limit_key]:
            raise RuntimeError('Cumulative resource envelope exceeds '+limit_key)
    if old['artifact_bytes_conservative']+caps['artifact_bytes']>limits['artifact_bytes']:
        raise RuntimeError('Cumulative artifact envelope exceeded')

def reject_duplicate_or_wrong_order(completed,remaining,matrix):
    k=lambda r:(r['parent'],r['repeat'],r['arm'])
    c=[k(r) for r in completed];r=[k(x) for x in remaining];p=[k(x) for x in matrix]
    if len(c)!=len(set(c)) or len(r)!=len(set(r)) or set(c)&set(r) or c+r!=p:
        raise RuntimeError('Recovery must use the exact frozen suffix')

def cumulative_snapshot(old_stages,segment):
    result={}
    for name,old in old_stages.items():
        v=segment['stages'].get(name,{})
        result[name]={s:old[s]+v.get(s,0) for s in ['reserved','verified','unknown']}
        result[name]['pending']=result[name]['reserved']-result[name]['verified']-result[name]['unknown']
        limit=old['limit']+(1 if name in ('C/model_loads','C/environment_resets') else 0)
        result[name]['prior_authorized_limit']=old['limit']
        result[name]['authorized_limit_including_explicit_extension']=limit
        if result[name]['reserved']>limit:raise RuntimeError('Cumulative stage cap exceeded')
    return result

PREFIX_FIELDS=('step','public','candidates','action','submit_command','diagnostic','vector_reward','scalar_reward','done','info')
def verify_prefix_step(actual,expected):
    # Timings, reservation tokens and process-lifetime counters necessarily differ.
    # No behavioral field is compared using a numeric tolerance.
    canonical=lambda value:json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
    for field in PREFIX_FIELDS:
        if canonical(actual[field])!=canonical(expected[field]):raise RuntimeError('Partial prefix mismatch at '+field)
    if actual['done']:raise RuntimeError('Saved prefix unexpectedly terminates')
