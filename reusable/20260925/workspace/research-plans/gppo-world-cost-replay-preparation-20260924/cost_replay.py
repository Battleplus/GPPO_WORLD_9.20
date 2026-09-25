"""Explicitly authorized, fixed-trace CPU diagnostic; never constructs an environment."""
from pathlib import Path
import argparse
import ctypes
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
PREP=ROOT.parent/'gppo-world-stage2-gate-preparation-20260924'
sys.path.insert(0,str(PREP))
from public_controller import PublicMemory,PublicPlanner,public_copy
from run_gate import peak_rss

REPETITIONS=21  # First pass reported cold; 20 fixed additional passes, all budgeted.
LIMITS={'environment_steps':0,'policy_encode':525,'world_candidate_batch':525,
        'actor_readout':525,'rule_decisions':651}

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def authorize(path):
    if path is None or not Path(path).is_file():raise PermissionError('New cost-replay approval required')
    auth=read(path);binding=read(ROOT/'binding.json')
    if (auth.get('approved') is not True or auth.get('stage')!='fixed_trace_cost_diagnostic'
        or auth.get('limits')!=LIMITS or auth.get('binding_sha256')!=sha(ROOT/'binding.json')
        or not auth.get('user_approval_reference') or not auth.get('user_approval_text')
        or auth.get('formal_stage_authorized') is not False or auth.get('updates')!=0
        or auth.get('wall_seconds_max')!=600 or auth.get('rss_bytes_max')!=4*1024**3
        or auth.get('artifact_bytes_max')!=512*1024**2):raise PermissionError('Unbound approval')
    for row in binding['inputs']:
        if sha(row['path'])!=row['sha256']:raise RuntimeError('Input changed: '+row['path'])
    return auth,binding

def groups(rows,arm):
    result=[]
    for parent in ['gate-000','gate-001']:
        group=sorted((r for r in rows if r['arm']==arm and r['parent']==parent),key=lambda r:r['step'])
        if not group or [r['step'] for r in group]!=list(range(len(group))):raise ValueError('Incomplete trace')
        result.append(group)
    if sum(map(len,result))!=({'M':25,'R':31}[arm]):raise ValueError('Unexpected trace population')
    return result

def replay_batch(traces,choose,clock_wall=time.perf_counter,clock_cpu=time.process_time):
    """No file IO, SQL, hashing or reward/true-state access inside timed block."""
    records=[];count=0;wall0=clock_wall();cpu0=clock_cpu()
    for episode in traces:
        memory=PublicMemory();state=None
        for row in episode:
            started=clock_wall()
            public=public_copy(row['observation']);memory.observe(public)
            candidates=memory.candidates(public)
            action,state,hidden=choose(public,memory,state,candidates)
            memory.submitted(public,action)
            elapsed=clock_wall()-started
            records.append((row['parent'],row['step'],action,hidden,elapsed));count+=1
    cpu=clock_cpu()-cpu0;wall=clock_wall()-wall0
    if cpu<0 or wall<=0:raise RuntimeError('Invalid CPU/wall counter')
    return {'decisions':count,'cpu_seconds':cpu,'wall_seconds':wall},records

def verify_records(traces,records,hidden_digest=None):
    expected={(r['parent'],r['step']):r for ep in traces for r in ep}
    if len(expected)!=len(records):raise RuntimeError('Output count mismatch')
    for parent,step,action,hidden,elapsed in records:
        row=expected[(parent,step)]
        if action!=row['action']:raise RuntimeError('Action reproduction mismatch')
        if hidden_digest is not None:
            if hidden_digest(hidden)!=row['diagnostic']['selected_world_hidden_sha256']:
                raise RuntimeError('Hidden reproduction mismatch')

def process_cpu_after_exit(process):
    # Read the exited child's retained process handle; includes startup and output writes.
    class FileTime(ctypes.Structure):_fields_=[('low',ctypes.c_ulong),('high',ctypes.c_ulong)]
    values=[FileTime() for _ in range(4)]
    fn=ctypes.windll.kernel32.GetProcessTimes
    fn.argtypes=[ctypes.c_void_p]+[ctypes.POINTER(FileTime)]*4;fn.restype=ctypes.c_int
    if not fn(ctypes.c_void_p(int(process._handle)),*(ctypes.byref(v) for v in values)):
        raise OSError('Cannot read completed child CPU')
    return sum((v.high<<32)+v.low for v in values[2:])*1e-7

def worker(arm,auth_path):
    auth,binding=authorize(auth_path);out=ROOT/'runs/cost-diagnostic-v1'
    permit=read(out/(arm+'-permit.json'))
    if permit['arm']!=arm:raise PermissionError('Wrong worker permit')
    save(out/(arm+'-claim.json'),{'pid':os.getpid(),'arm':arm})
    worker_start=time.perf_counter()
    all_rows=[json.loads(s) for s in Path(binding['decisions']).read_text(encoding='utf-8').splitlines()]
    traces=groups(all_rows,arm);counts={k:0 for k in LIMITS};hidden_digest=None
    planner=PublicPlanner()
    if arm=='M':
        native=read(PREP/'binding.json');src=Path(native['native_root'])
        sys.path.insert(0,str(src))
        import torch
        if torch.__version__!='2.13.0+cpu' or tuple(sys.version_info[:3])!=(3,14,4):raise RuntimeError('Runtime mismatch')
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        from gppo_world.m10_environment import M10Config
        from gppo_world.joint_gppo import JointGraphPreferencePolicy,ActionConditionedTemporalWorldModel,masked_normalized_preference
        for name,module in list(sys.modules.items()):
            if name=='gppo_world' or name.startswith('gppo_world.'):
                path=Path(module.__file__).resolve();relative=path.relative_to(src).as_posix()
                if sha(path)!=native['native_python_files'].get(relative):raise RuntimeError('Native import mismatch')
        config=M10Config(**json.loads(Path(native['config']).read_text(encoding='utf-8-sig'))['environment'])
        policy=JointGraphPreferencePolicy(config,history=True);world=ActionConditionedTemporalWorldModel()
        payload=torch.load(native['checkpoint'],map_location='cpu',weights_only=False)
        if payload.get('group')!='P_train':raise RuntimeError('Checkpoint group mismatch')
        policy.load_state_dict(payload['policy_state_dict'],strict=True);world.load_state_dict(payload['world_state_dict'],strict=True)
        for m in [policy,world]:
            m.eval()
            for p in m.parameters():p.requires_grad_(False)
        preference=masked_normalized_preference((.8,.2),device=torch.device('cpu'))
        def state_hash():
            h=hashlib.sha256()
            for tag,m in [('policy',policy),('world',world)]:
                for k,v in sorted(m.state_dict().items()):h.update((tag+k).encode());h.update(v.detach().cpu().numpy().tobytes())
            return h.hexdigest()
        before=state_hash()
        def choose(public,memory,state,candidates):
            ph,wh=(None,None) if state is None else state
            with torch.no_grad():
                counts['policy_encode']+=1
                f,pair,nph=policy.encode(torch.as_tensor(public['flat'],dtype=torch.float32).reshape(1,-1),ph)
                counts['world_candidate_batch']+=1
                features,by_action=world.predict_all_candidates(f,wh,public,use_events=True)
                counts['actor_readout']+=1
                evaluation=policy.evaluate_encoded(f,pair,preference,features,torch.as_tensor(public['mask'],dtype=torch.bool)[None,:])
                probs=evaluation['probabilities'][0].detach().cpu().tolist()
                if not all(math.isfinite(v) for v in probs):raise RuntimeError('Nonfinite actor probabilities')
                action=max(candidates,key=lambda a:(probs[a],-a))
                hidden=by_action[action]['hidden'].detach()
                return action,(nph.detach(),hidden),hidden
        hidden_digest=lambda h:hashlib.sha256(h.cpu().numpy().tobytes()).hexdigest()
    else:
        if 'torch' in sys.modules:raise RuntimeError('Rule process contaminated by Torch')
        def choose(public,memory,state,candidates):
            counts['rule_decisions']+=1
            action,_=planner.choose(public,memory)
            return action,None,None
    setup_seconds=time.perf_counter()-worker_start;batches=[]
    for index in range(REPETITIONS):
        if time.perf_counter()-worker_start>550 or peak_rss()>4*1024**3:raise RuntimeError('Resource cap')
        stats,records=replay_batch(traces,choose)
        verify_records(traces,records,hidden_digest)
        stats.update({'index':index,'cold':index==0,'decision_wall_seconds':[r[4] for r in records]})
        batches.append(stats)
    if arm=='M' and state_hash()!=before:raise RuntimeError('Model weights changed')
    if peak_rss()>4*1024**3:raise RuntimeError('Final memory cap')
    save(out/(arm+'-result.json'),{'arm':arm,'counts':counts,'batches':batches,
        'setup_wall_seconds':setup_seconds,'peak_rss_bytes':peak_rss(),
        'action_reproduction':True,'hidden_reproduction':arm=='M','updates':0,
        'measurement_scope':'Saved branch inputs only; no environment, new outcomes or benefit evidence.'})

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--authorization',type=Path);parser.add_argument('--worker',choices=['M','R'])
    args=parser.parse_args(argv)
    if args.worker:return worker(args.worker,args.authorization)
    auth,binding=authorize(args.authorization);out=ROOT/'runs/cost-diagnostic-v1';out.mkdir(parents=True,exist_ok=False)
    save(out/'authorization.json',auth);save(out/'binding.json',binding)
    budget_path=Path(binding['budget_executor'])
    spec=importlib.util.spec_from_file_location('pinned_budget_executor',budget_path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    budget=mod.PersistentBudget(out/'budget.sqlite3',limits=LIMITS,run_limits=LIMITS,attempt_id='cost-diagnostic-v1-once',run_id='cost-diagnostic-v1')
    started=time.perf_counter();tokens=[];workers=[]
    try:
        for arm in ['M','R']:
            amounts={k:v for k,v in LIMITS.items() if v and ((arm=='M' and k!='rule_decisions') or (arm=='R' and k=='rule_decisions'))}
            tokens=[budget.reserve(k,n) for k,n in amounts.items()]
            save(out/(arm+'-permit.json'),{'arm':arm,'reservations':tokens})
            with (out/(arm+'-console.txt')).open('x',encoding='utf-8') as f:
                launch=time.perf_counter()
                child=subprocess.Popen([sys.executable,'-B','-X','utf8',str(Path(__file__).resolve()),'--worker',arm,'--authorization',str(args.authorization.resolve())],stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                try:code=child.wait(timeout=max(.1,600-(time.perf_counter()-started)))
                except subprocess.TimeoutExpired:child.kill();child.wait();raise RuntimeError('Worker time cap; no retry')
                workers.append({'arm':arm,'process_cpu_seconds':process_cpu_after_exit(child),'lifetime_wall_seconds':time.perf_counter()-launch,'exit_code':code})
            if code!=0:raise RuntimeError('Worker failed; reservations remain consumed/unknown')
            result=read(out/(arm+'-result.json'))
            if any(result['counts'][k]!=n for k,n in amounts.items()):raise RuntimeError('Counter mismatch')
            for token in tokens:budget.complete(token)
            tokens=[]
            if sum(p.stat().st_size for p in out.iterdir() if p.is_file())>512*1024**2:raise RuntimeError('Artifact cap')
        save(out/'status.json',{'status':'completed','workers':workers,'budget':budget.snapshot(),'formal_stage_authorized':False,'updates':0})
    except BaseException as exc:
        for token in tokens:
            try:budget.unknown(token,str(exc))
            except Exception:pass
        save(out/'status.json',{'status':'technical_stop','error':str(exc),'workers':workers,'budget':budget.snapshot(),'automatic_retry':False})
        raise

if __name__=='__main__':
    try:main()
    except Exception as exc:print(type(exc).__name__+': '+str(exc));sys.exit(2)
