"""One approved conditional A -> B -> C attempt; no automatic retries."""
import argparse
import queue
import subprocess
import threading

from runtime_support import *
PROGRAM_STARTED=time.perf_counter()

MODEL_KEYS=('policy_encode','world_candidate_batch','actor_readout')

def limits_for(request):
    limits={}
    for s,row in request['stage_caps'].items():
        for k in MODEL_KEYS:limits[s+'/'+k]=row['each_model_forward']
        limits[s+'/environment_steps']=row['environment_steps']
        limits[s+'/rule_decisions']=row.get('rule_decisions',0)
        limits[s+'/environment_resets']={'A':0,'B':4,'C':9600}[s]
        limits[s+'/model_loads']=2 if s=='A' else 1
        for k in ['optimizer_updates','world_updates','offline_updates']:limits[s+'/'+k]=0
    return limits

class Monitor:
    def __init__(self,request,out):
        self.request=request;self.out=out;self.children=[];self.started=PROGRAM_STARTED;self.parent_start=0.
        self.parent_parts={};self.peak_sum=0;self.enter('A');self.stage_cpu_start=0.
    def enter(self,stage):
        self.stage=stage;self.stage_start=time.perf_counter();self.stage_cpu_start=self.total_cpu();self.stage_parent_cpu_start=time.process_time();self.stage_disk_start=self.disk()
    def disk(self):return sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file())
    def total_cpu(self):return time.process_time()-self.parent_start+sum(process_cpu(c.p._handle) for c in self.children)
    def measure(self,category,fn,*args,**kwargs):
        value,elapsed=timed(fn,*args,**kwargs)
        part=self.parent_parts.setdefault(self.stage+'/'+category,{'cpu':0.,'wall':0.})
        for k,v in elapsed.items():part[k]+=v
        return value
    def check(self,disk=False):
        cap=self.request['stage_caps'][self.stage];cpu=self.total_cpu()
        if time.perf_counter()-self.started>self.request['caps']['wall_seconds'] or time.perf_counter()-self.stage_start>cap['wall_seconds']:raise RuntimeError('Wall cap')
        if cpu>self.request['caps']['total_process_cpu_seconds'] or cpu-self.stage_cpu_start>cap['cpu_seconds']:raise RuntimeError('CPU cap')
        # Conservative sum: parent peak plus every resident child peak, including suspended workers.
        rss=peak_rss()
        for c in self.children:
            if c.p.poll() is None:
                try:rss+=process_memory(c.p._handle)[1]
                except OSError:
                    if c.p.poll() is None:raise
        self.peak_sum=max(self.peak_sum,rss)
        if rss>self.request['caps']['resident_process_rss_sum_bytes']:raise RuntimeError('Combined RSS cap')
        if disk:
            size=self.disk();stage_cap=(1 if self.stage in ('A','B') else 40)*1024**3
            if size>self.request['caps']['artifact_bytes'] or size-self.stage_disk_start>stage_cap:raise RuntimeError('Artifact cap')
    def snapshot(self):
        return {'wall_seconds':time.perf_counter()-self.started,'total_cpu_seconds':self.total_cpu(),
            'current_stage':self.stage,'stage_wall_seconds':time.perf_counter()-self.stage_start,
            'stage_cpu_seconds':self.total_cpu()-self.stage_cpu_start,'stage_parent_cpu_seconds':time.process_time()-self.stage_parent_cpu_start,
            'parent_cpu_seconds':time.process_time()-self.parent_start,'parent_timed_parts':self.parent_parts,
            'peak_conservative_rss_sum':self.peak_sum,'rss_scope':'parent plus all resident workers, including suspended; sum of individual peaks','bytes':self.disk(),
            'workers':[{'name':c.name,'cpu_seconds':process_cpu(c.p._handle),'exit_code':c.p.poll(),'ready':getattr(c,'ready',None),'final':getattr(c,'final',None)} for c in self.children]}

class OwnedWorker:
    def __init__(self,name,stage,kind,auth,monitor):
        self.name=name;self.monitor=monitor;self.suspended=False;self.responses=queue.Queue();out=monitor.out
        write(out/(name+'-permit.json'),{'stage':stage,'kind':kind})
        self.err=(out/(name+'-stderr.txt')).open('x',encoding='utf-8')
        self.p=subprocess.Popen([sys.executable,'-B','-X','utf8',str(ROOT/'stage_worker.py'),'--authorization',str(auth.resolve()),'--name',name],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.err,text=True,encoding='utf-8',bufsize=1,creationflags=subprocess.CREATE_NO_WINDOW)
        monitor.children.append(self)
        def reader():
            try:
                for line in self.p.stdout:self.responses.put(json.loads(line))
                self.responses.put({'ok':False,'error':'Worker stdout closed'})
            except BaseException as exc:self.responses.put({'ok':False,'error':str(exc)})
        self.reader=threading.Thread(target=reader,daemon=True);self.reader.start()
        self.ready=self.receive();self.pause()
    def receive(self):
        while True:
            self.monitor.check()
            try:response=self.responses.get(timeout=.2)
            except queue.Empty:continue
            if not response.get('ok'):raise RuntimeError('Worker failure: '+str(response))
            return response
    def pause(self):
        if self.p.poll() is None:suspend(self.p._handle);self.suspended=True
    def resume(self):
        if self.suspended:suspend(self.p._handle,resume=True);self.suspended=False
    def call(self,command,**kwargs):
        self.resume();self.p.stdin.write(canonical({'command':command,**kwargs})+'\n');self.p.stdin.flush()
        response=self.receive();self.pause();return response['result']
    def close(self):
        self.resume();self.p.stdin.write(canonical({'command':'shutdown'})+'\n');self.p.stdin.flush()
        self.final=self.receive()['result'];self.p.stdin.close()
        while self.p.poll() is None:
            self.monitor.check()
            try:self.p.wait(timeout=.2)
            except subprocess.TimeoutExpired:continue
        if self.p.returncode!=0:raise RuntimeError('Worker nonzero exit')
        self.err.close()
    def kill(self):
        if self.p.poll() is None:self.p.kill();self.p.wait(timeout=10)
        self.err.close()

class Reservations:
    def __init__(self,budget,monitor):self.budget=budget;self.monitor=monitor;self.pending=[]
    def reserve(self,amounts):
        if self.pending:raise RuntimeError('Unresolved prior reservation')
        for key,amount in amounts.items():
            if amount:self.pending.append(self.monitor.measure('ledger',self.budget.reserve,self.monitor.stage+'/'+key,amount))
    def complete(self):
        for token in list(self.pending):self.monitor.measure('ledger',self.budget.complete,token);self.pending.remove(token)
    def unknown(self,error):
        for token in list(self.pending):
            try:self.budget.unknown(token,str(error))
            except Exception:pass

def summarize_replay(result):
    warm=result['batches'][1:];latencies=sorted(t for b in warm for t in b['decision_wall_seconds'])
    p=(len(latencies)-1)*.95;i=int(p);q=latencies[i]+(latencies[min(i+1,len(latencies)-1)]-latencies[i])*(p-i)
    cpu=sum(b['cpu_seconds'] for b in warm)/sum(b['decisions'] for b in warm)
    return {'warm_cpu_seconds_per_decision':cpu,'warm_wall_p95_seconds':q,
        'passed':bool(result['equivalence'] and result['weights_unchanged'] and cpu<=.010 and q<=.050 and result['peak_rss']<=4*1024**3)}

def stage_a(auth,monitor,reservations):
    (monitor.out/'A').mkdir()
    results={}
    for kind in ['reference','candidate']:
        reservations.reserve({'model_loads':1});worker=OwnedWorker('A-'+kind,'A',kind,auth,monitor)
        if worker.ready['counts']['model_loads']!=1:raise RuntimeError('Model load accounting mismatch')
        reservations.complete();reservations.reserve({k:525 for k in MODEL_KEYS})
        result=worker.call('replay')
        if any(result['counts'][k]!=525 for k in MODEL_KEYS):raise RuntimeError('Replay forward count mismatch')
        monitor.measure('logging',write,monitor.out/('A/'+kind+'-summary.json'),summarize_replay(result))
        reservations.complete();worker.close();monitor.check(disk=True);results[kind]=summarize_replay(result)
    write(monitor.out/'A/decision.json',results)
    return results['candidate']['passed']

def stages_environment(stage,auth,monitor,reservations,native):
    monitor.enter(stage);folder=monitor.out/stage;folder.mkdir()
    count=2 if stage=='B' else 1600;seed=924110000 if stage=='B' else 924200000;repeats=1 if stage=='B' else 3
    rules=OwnedWorker(stage+'-R',stage,'rule',auth,monitor)
    if rules.ready['torch_loaded']:raise RuntimeError('Rule process has Torch')
    tapes=rules.call('generate',count=count,seed=seed)
    if len(tapes['tapes'])!=count or any(digest(t)!=h for t,h in zip(tapes['tapes'],tapes['hashes'])):raise RuntimeError('Scenario manifest mismatch')
    if any(t['seed'] in set(native['historical_seed_ids']) for t in tapes['tapes']):raise RuntimeError('Historical seed collision')
    if [t['seed'] for t in tapes['tapes']]!=list(range(seed,seed+count)):raise RuntimeError('Wrong planned seeds')
    write(folder/'scenario-manifest.json',tapes)
    reservations.reserve({'model_loads':1});models=OwnedWorker(stage+'-M',stage,'candidate',auth,monitor)
    if models.ready['counts']['model_loads']!=1:raise RuntimeError('Model load accounting mismatch')
    reservations.complete()
    logger=CompressedRecords(folder/'steps.jsonl.gz');episodes=[];latencies={'M':[],'R':[]};part_totals={};message_counts={};interface=0
    try:
        for index,tape in enumerate(tapes['tapes']):
            parent=(f'new-gate-{index:03d}' if stage=='B' else f'eval-{index:04d}')
            for repeat in range(repeats):
                initials={}
                key=(f'stage2-decision-only-v1|{parent}|repeat-{repeat}' if stage=='B'
                    else f'frozen-native-public-history-allocation-v1|{parent}|repeat-{repeat}')
                for arm in (['M','R'] if (index+repeat)%2==0 else ['R','M']):
                    worker=models if arm=='M' else rules;before=worker.ready['counts'].copy() if not hasattr(worker,'last_counts') else worker.last_counts.copy()
                    reservations.reserve({'environment_resets':1});reset=worker.call('reset',tape=tape,key=key)
                    if reset['counts']['environment_resets']!=before['environment_resets']+1:raise RuntimeError('Reset count mismatch')
                    worker.last_counts=reset['counts'];initials[arm]=reset['initial_hash'];memory=PublicMemory()
                    monitor.measure('logging',append,folder/'episode-starts.jsonl',{'parent':parent,'repeat':repeat,'arm':arm,'random_key':key,'tape_hash':digest(tape),'reset':reset})
                    reservations.complete();uc=0.;prevcounts={'completed':0,'expired':0};energy=36.
                    for step in range(18):
                        amounts={'environment_steps':1,**({k:1 for k in MODEL_KEYS} if arm=='M' else {'rule_decisions':1})}
                        reservations.reserve(amounts);row=worker.call('step')
                        if row['step']!=step:raise RuntimeError('Step ordering mismatch')
                        for k,n in amounts.items():
                            if row['counts'][k]-worker.last_counts[k]!=n:raise RuntimeError('Step call counter mismatch')
                        worker.last_counts=row['counts'];public=row['public'];memory.observe(public)
                        if list(memory.candidates(public))!=row['candidates'] or row['action'] not in memory.candidates(public):raise RuntimeError('Public guard mismatch')
                        if not public['mask'][row['action']]:raise RuntimeError('Illegal action')
                        memory.submitted(public,row['action']);info=row['info']
                        import struct
                        f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
                        en=sum(info['energy'].values());vec=[f32((max(0,info['counts']['completed']-prevcounts['completed'])-max(0,info['counts']['expired']-prevcounts['expired']))/6),f32(-max(0,energy-en)/36)]
                        if vec!=row['vector_reward']:raise RuntimeError('Independent reward mismatch')
                        prevcounts,energy=info['counts'],en;uc+=.99**step*(.4*vec[0]+.2*vec[1]);ending(row['done'],info)
                        row.update(parent=parent,repeat=repeat,arm=arm,reservations=list(reservations.pending))
                        monitor.measure('logging',logger.add,row);reservations.complete();interface+=1
                        latencies[arm].append(row['controller'])
                        for label in ['controller','environment','labels']:
                            d=part_totals.setdefault(arm+'/'+label,{'cpu':0.,'wall':0.})
                            for k,v in row[label].items():d[k]+=v
                        for event in info['communication_delta']:
                            k=f"{arm}/{event['link']}/{event['status']}";message_counts[k]=message_counts.get(k,0)+1
                        if row['done']:
                            ep=row['episode'];ep.update(parent=parent,repeat=repeat,arm=arm,initial_public_hash=reset['initial_hash'])
                            if abs(ep['utility']-uc)>1e-12:raise RuntimeError('Episode utility mismatch')
                            monitor.measure('logging',append,folder/'episodes.jsonl',ep);episodes.append(ep);break
                    else:raise RuntimeError('No native episode end')
                    monitor.check(disk=True)
                if initials['M']!=initials['R']:raise RuntimeError('Paired reset mismatch')
        logger.close();verify_compressed(folder/'steps.jsonl.gz',logger.digest.hexdigest(),logger.rows)
        models.close();rules.close()
        runtime={'latencies':latencies,'worker_parts':part_totals,'communication_events':message_counts,
            'compressed_digest':logger.digest.hexdigest(),'compressed_rows':logger.rows,'interface_steps':interface,'resources':monitor.snapshot()}
        runtime['unclassified_worker_cpu']={}
        for arm in ['M','R']:
            w=next(w for w in runtime['resources']['workers'] if w['name']==stage+'-'+arm)
            explained=w['ready']['startup_cpu']+sum(v['cpu'] for k,v in part_totals.items() if k.startswith(arm+'/'))
            residual=w['cpu_seconds']-explained
            if residual < -1e-8:raise RuntimeError('Overlapping worker CPU attribution')
            runtime['unclassified_worker_cpu'][arm]=max(0.,residual)
        if not all(sum(t['cpu'] for t in latencies[a])>0 for a in ['M','R']):raise RuntimeError('Insufficient aggregate controller CPU recording')
        write(folder/'cost-and-interface.json',runtime);monitor.check(disk=True)
        if len(episodes)!=count*repeats*2:raise RuntimeError('Incomplete matrix')
        write(folder/'status.json',{'status':'completed','episodes':len(episodes),'updates':0,'cost_recording':'controller/environment/labels + parent ledger/logging + complete process CPU; residual explicitly unclassified'})
        return episodes,runtime
    finally:
        if not logger.raw.closed:logger.close()

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--authorization',type=Path);args=parser.parse_args(argv)
    authorization,binding,request=authorize(args.authorization)
    if os.name!='nt':raise RuntimeError('Frozen Windows runtime required')
    out=ROOT/'runs/consolidated-v1';out.mkdir(parents=True,exist_ok=False)
    write(out/'authorization.json',authorization);write(out/'execution-manifest.json',binding)
    native=read(PREP/'binding.json');monitor=Monitor(request,out)
    budget=budget_class(native)(out/'budget.sqlite3',limits=limits_for(request),run_limits=limits_for(request),attempt_id='consolidated-v1-once',run_id='consolidated-v1')
    reservations=Reservations(budget,monitor)
    try:
        if not stage_a(args.authorization,monitor,reservations):
            write(out/'status.json',{'status':'research_stop_A_cost','budget':budget.snapshot(),'resources':monitor.snapshot(),'B_and_C_not_started':True});return
        stages_environment('B',args.authorization,monitor,reservations,native)
        # The new scenario manifests and stage accounting remain separately identified.
        episodes,cost=stages_environment('C',args.authorization,monitor,reservations,native)
        from analyze_formal import analyze
        result=analyze(episodes,cost);write(out/'C/analysis.json',result)
        snapshot=budget.snapshot()
        if any(v['pending'] or v['unknown'] for v in snapshot['stages'].values()):raise RuntimeError('Unresolved budget')
        authorize(args.authorization);monitor.check(disk=True)
        write(out/'status.json',{'status':'completed','decision':result['decision'],'budget':snapshot,'resources':monitor.snapshot(),'updates':0,'stages3_to5_authorized':False})
    except BaseException as exc:
        reservations.unknown(exc)
        for c in monitor.children:
            try:c.kill()
            except Exception:pass
        write(out/'status.json',{'status':'technical_stop','error':f'{type(exc).__name__}: {exc}','budget':budget.snapshot(),'resources':monitor.snapshot(),'automatic_retry':False})
        raise

if __name__=='__main__':
    try:main()
    except Exception as exc:print(f'{type(exc).__name__}: {exc}');sys.exit(2)
