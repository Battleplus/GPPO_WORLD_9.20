"""One approved conditional A -> B -> C attempt; no automatic retries."""
import argparse
import queue
import subprocess
import threading

from runtime_support import *
PROGRAM_STARTED=time.perf_counter()
from recovery_contract import cumulative_snapshot, reject_duplicate_or_wrong_order, verify_prefix_step
from sqlite_diagnostics import call_once
from session_budget import single_writer_budget
import traceback
OLD=ROOT.parent
OLD_RUN=OLD/'runs/consolidated-v1'

MODEL_KEYS=('policy_encode','world_candidate_batch','actor_readout')

def limits_for(request):
    return {'C/'+k:request['caps'][k] for k in [*MODEL_KEYS,'environment_steps','rule_decisions','environment_resets','model_loads','optimizer_updates','world_updates','offline_updates']}


class Monitor:
    def __init__(self,request,out):
        self.request=request;self.out=out;self.children=[];self.started=PROGRAM_STARTED;self.parent_start=0.
        self.parent_parts={};self.peak_sum=0;self.enter('C');self.stage_cpu_start=0.
    def enter(self,stage):
        self.stage=stage;self.stage_start=time.perf_counter();self.stage_cpu_start=self.total_cpu();self.stage_parent_cpu_start=time.process_time();self.stage_disk_start=self.disk()
    def disk(self):return sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
    def total_cpu(self):return time.process_time()-self.parent_start+sum(process_cpu(c.p._handle) for c in self.children)
    def measure(self,category,fn,*args,**kwargs):
        value,elapsed=timed(fn,*args,**kwargs)
        part=self.parent_parts.setdefault(self.stage+'/'+category,{'cpu':0.,'wall':0.})
        for k,v in elapsed.items():part[k]+=v
        return value
    def check(self,disk=False):
        cap=self.request['stage_caps'][self.stage];cpu=self.total_cpu();hold=self.request['recovery_shutdown_holdback']
        if time.perf_counter()-self.started>self.request['caps']['wall_seconds']-hold['wall_seconds'] or time.perf_counter()-self.stage_start>cap['wall_seconds']-hold['wall_seconds']:raise RuntimeError('Wall cap')
        if cpu>self.request['caps']['total_process_cpu_seconds']-hold['cpu_seconds'] or cpu-self.stage_cpu_start>cap['cpu_seconds']-hold['cpu_seconds']:raise RuntimeError('CPU cap')
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
            size=self.disk();stage_cap=self.request['caps']['artifact_bytes']
            if size>self.request['caps']['artifact_bytes'] or size>stage_cap:raise RuntimeError('Artifact cap')
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
            if amount:self.pending.append(self.monitor.measure('ledger',call_once,'reserve',self.budget.reserve,self.monitor.stage+'/'+key,amount,diagnostic_dir=self.monitor.out/'diagnostics',db_path=self.budget.db_path))
    def complete(self):
        for token in list(self.pending):
            self.monitor.measure('ledger',call_once,'complete',self.budget.complete,token,diagnostic_dir=self.monitor.out/'diagnostics',db_path=self.budget.db_path)
            self.pending.remove(token)
    def unknown(self,error):
        for token in list(self.pending):
            try:call_once('mark_unknown',self.budget.unknown,token,str(error),diagnostic_dir=self.monitor.out/'diagnostics',db_path=self.budget.db_path)
            except Exception as secondary:error.add_note('Unable to mark pending reservation unknown: '+repr(secondary))

def stages_environment(stage,auth,monitor,reservations,native):
    monitor.enter(stage);folder=monitor.out/stage;folder.mkdir()
    if stage!='C':raise RuntimeError('Recovery only resumes C; A/B are evidence, not rerun')
    count=1600;repeats=3
    retained=read(ROOT/'retained-episodes.json');remaining=read(ROOT/'not-executed.json');matrix=read(ROOT/'frozen-matrix.json')
    reject_duplicate_or_wrong_order(retained,remaining,matrix)
    completed={(r['parent'],r['repeat'],r['arm']) for r in retained}
    old_initials={(r['parent'],r['repeat'],r['arm']):r['initial_public_hash'] for r in retained}
    remaining_keys=[(r['parent'],r['repeat'],r['arm']) for r in remaining];executed=[]
    prefix=read(ROOT/'partial-prefix.json');prefix_key=tuple(prefix['key']);prefix_verified=0
    if prefix_key!=remaining_keys[0] or len(prefix['steps'])!=7:raise RuntimeError('Wrong partial prefix identity')
    rules=OwnedWorker('C-R',stage,'rule',auth,monitor)
    if rules.ready['torch_loaded']:raise RuntimeError('Rule process has Torch')
    tapes=read(OLD_RUN/'C/scenario-manifest.json')
    if len(tapes['tapes'])!=1600 or any(digest(t)!=h for t,h in zip(tapes['tapes'],tapes['hashes'])):raise RuntimeError('Scenario identity mismatch')
    write(folder/'scenario-manifest.json',tapes)
    reservations.reserve({'model_loads':1});models=OwnedWorker(stage+'-M',stage,'candidate',auth,monitor)
    if models.ready['counts']['model_loads']!=1:raise RuntimeError('Model load accounting mismatch')
    reservations.complete()
    logger=CompressedRecords(folder/'steps.jsonl.gz');episodes=[];latencies={'M':[],'R':[]};replay_latencies=[];part_totals={};message_counts={};interface=0
    try:
        for index,tape in enumerate(tapes['tapes']):
            parent=f'eval-{index:04d}'
            for repeat in range(repeats):
                initials={a:old_initials[(parent,repeat,a)] for a in ['M','R'] if (parent,repeat,a) in completed}
                key=(f'stage2-decision-only-v1|{parent}|repeat-{repeat}' if stage=='B'
                    else f'frozen-native-public-history-allocation-v1|{parent}|repeat-{repeat}')
                for arm in (['M','R'] if (index+repeat)%2==0 else ['R','M']):
                    episode_key=(parent,repeat,arm)
                    if episode_key in completed:continue
                    if episode_key!=remaining_keys[len(executed)]:raise RuntimeError('Recovery order changed')
                    worker=models if arm=='M' else rules;before=worker.ready['counts'].copy() if not hasattr(worker,'last_counts') else worker.last_counts.copy()
                    reservations.reserve({'environment_resets':1});reset=worker.call('reset',tape=tape,key=key)
                    if reset['counts']['environment_resets']!=before['environment_resets']+1:raise RuntimeError('Reset count mismatch')
                    worker.last_counts=reset['counts'];initials[arm]=reset['initial_hash'];memory=PublicMemory()
                    if episode_key==prefix_key and reset['initial_hash']!=prefix['reset']['reset']['initial_hash']:raise RuntimeError('Partial reset identity mismatch')
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
                        if episode_key==prefix_key and step<7:
                            verify_prefix_step(row,prefix['steps'][step]);prefix_verified+=1;replay_latencies.append(row['controller'])
                            if prefix_verified==7:write(folder/'prefix-reproduction.json',{'key':prefix['key'],'verified_steps':7,'exact_behavior_match':True,'counts_charged':True,'new_statistical_samples':0})
                        else:latencies[arm].append(row['controller'])
                        for label in ['controller','environment','labels']:
                            d=part_totals.setdefault(arm+'/'+label,{'cpu':0.,'wall':0.})
                            for k,v in row[label].items():d[k]+=v
                        for event in info['communication_delta']:
                            k=f"{arm}/{event['link']}/{event['status']}";message_counts[k]=message_counts.get(k,0)+1
                        if row['done']:
                            ep=row['episode'];ep.update(parent=parent,repeat=repeat,arm=arm,initial_public_hash=reset['initial_hash'])
                            if abs(ep['utility']-uc)>1e-12:raise RuntimeError('Episode utility mismatch')
                            monitor.measure('logging',append,folder/'episodes.jsonl',ep);episodes.append(ep);executed.append(episode_key);publish_progress(monitor,len(executed),episode_key);break
                    else:raise RuntimeError('No native episode end')
                    monitor.check(disk=True)
                if initials['M']!=initials['R']:raise RuntimeError('Paired reset mismatch')
        logger.close();verify_compressed(folder/'steps.jsonl.gz',logger.digest.hexdigest(),logger.rows)
        models.close();rules.close()
        if prefix_verified!=7:raise RuntimeError('Partial prefix not fully reproduced')
        runtime={'latencies':latencies,'replay_overhead_latencies':replay_latencies,'replayed_steps':7,'worker_parts':part_totals,'communication_events':message_counts,
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
        if executed!=remaining_keys or len(episodes)!=963:raise RuntimeError('Incomplete recovery suffix')
        write(folder/'status.json',{'status':'completed','episodes':len(episodes),'updates':0,'cost_recording':'controller/environment/labels + parent ledger/logging + complete process CPU; residual explicitly unclassified'})
        return episodes,runtime
    finally:
        if not logger.raw.closed:logger.close()

def publish_progress(monitor, completed, key):
    # Observers only read this disposable status file, never live experiment logs.
    # Their failure cannot make an otherwise valid episode fail.
    path=monitor.out/'observer-progress.jsonl'
    try:
        with path.open('a',encoding='utf-8') as f:
            f.write(canonical({'recovery_completed':completed,'total_completed':8637+completed,'planned':9600,'last_key':key})+'\n')
            f.flush()
    except OSError:
        monitor.observer_publish_errors=getattr(monitor,'observer_publish_errors',0)+1

def merge_cost(old,new,resources):
    result={'latencies':{},'worker_parts':{},'communication_events':{},'resources':resources,
            'segmented_execution':True,'old_resources':old['resources'],'recovery_resources':new['resources'],
            'replay_overhead_latencies':new['replay_overhead_latencies'],'replayed_steps':7,
            'latency_rule':'Original 7 prefix timings retained; re-executed 7 timings reported as charged replay overhead'}
    for arm in ['M','R']:result['latencies'][arm]=old['latencies'][arm]+new['latencies'][arm]
    for source in [old,new]:
        for name,value in source['worker_parts'].items():
            target=result['worker_parts'].setdefault(name,{'cpu':0.,'wall':0.})
            for k,v in value.items():target[k]+=v
        for k,v in source['communication_events'].items():result['communication_events'][k]=result['communication_events'].get(k,0)+v
    return result

def cumulative_resources(monitor,request):
    value=monitor.snapshot();old=request['historical_resources']
    value['recovery_segment']=dict(value)
    value['wall_seconds']+=old['total_wall_seconds']
    value['total_cpu_seconds']+=old['total_cpu_seconds']
    value['peak_conservative_rss_sum']=max(value['peak_conservative_rss_sum'],old['peak_conservative_rss_sum'])
    value['bytes']+=old['artifact_bytes_conservative']
    value['historical_shutdown_unmeasured']=True
    value['guard_holdback_wall_seconds']=request['historical_shutdown_holdback']['wall_seconds']+request['recovery_shutdown_holdback']['wall_seconds']
    value['guard_holdback_cpu_seconds']=request['historical_shutdown_holdback']['cpu_seconds']+request['recovery_shutdown_holdback']['cpu_seconds']
    value['guard_charged_wall_seconds']=value['wall_seconds']+value['guard_holdback_wall_seconds']
    value['guard_charged_cpu_seconds']=value['total_cpu_seconds']+value['guard_holdback_cpu_seconds']
    value['not_exact_whole_os_lifecycle_measurement']=True
    return value

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--authorization',type=Path);args=parser.parse_args(argv)
    authorization,binding,request=authorize(args.authorization)
    if os.name!='nt':raise RuntimeError('Frozen Windows runtime required')
    out=ROOT/'runs/consolidated-v1';out.mkdir(parents=True,exist_ok=False)
    write(out/'authorization.json',authorization);write(out/'recovery-manifest.json',binding)
    native=read(PREP/'binding.json');monitor=Monitor(request,out)
    old_stages=read(ROOT/'historical-budget-readonly.json')
    histories=read(ROOT/'historical-ledgers-readonly.json')
    write(out/'ledger-lineage.json',{'prior_segments':histories,'historical_stages':old_stages,
        'additional_model_loads_explicitly_authorized':1,'additional_C_reset_explicitly_authorized':1,
        'historical_io_cause_unresolved':True,'approved_mitigation':'single-owned-connection-with-full-diagnostics',
        'cannot_refund_history':True})
    limits=limits_for(request);budget=None;reservations=None;snapshot=None;cumulative=None;ledger_closed=False
    try:
        Budget=single_writer_budget(budget_class(native))
        budget=call_once('initialize',Budget,out/'budget.sqlite3',limits=limits,run_limits=limits,
            attempt_id='consolidated-v1-recovery-2-local-once',run_id='consolidated-v1-recovery-2-local',
            diagnostic_dir=out/'diagnostics',db_path=out/'budget.sqlite3')
        reservations=Reservations(budget,monitor)
        episodes,cost=stages_environment('C',args.authorization,monitor,reservations,native)
        combined=read(ROOT/'retained-episodes.json')+episodes
        if len(combined)!=9600:raise RuntimeError('Original matrix incomplete')
        resources=cumulative_resources(monitor,request)
        merged=merge_cost(read(ROOT/'retained-cost-inputs.json'),cost,resources)
        from analyze_formal import analyze
        result=analyze(combined,merged)
        write(out/'C/combined-episodes.json',combined);write(out/'C/combined-cost.json',merged)
        write(out/'C/analysis.json',result)
        snapshot=call_once('snapshot',budget.snapshot,diagnostic_dir=out/'diagnostics',db_path=budget.db_path)
        cumulative=cumulative_snapshot(old_stages,snapshot)
        if any(v['pending'] or v['unknown'] for v in cumulative.values()):raise RuntimeError('Unresolved cumulative budget')
        call_once('close',budget.close,diagnostic_dir=out/'diagnostics',db_path=budget.db_path)
        ledger_closed=True
        authorize(args.authorization);monitor.check(disk=True)
        write(out/'status.json',{'status':'completed','decision':result['decision'],
            'segment_budget':snapshot,'cumulative_stages':cumulative,
            'resources':cumulative_resources(monitor,request),'updates':0,
            'observer_publish_errors':getattr(monitor,'observer_publish_errors',0),
            'replayed_steps':7,'ledger_connections':budget.connection_open_count,
            'historical_io_cause_unresolved':True,'stage3_to5_authorized':False,'automatic_retry':False})
    except BaseException as exc:
        primary_trace=traceback.format_exc()
        if reservations is not None:reservations.unknown(exc)
        for child in monitor.children:
            try:child.kill()
            except Exception:pass
        snapshot_error=None
        if budget is not None and not ledger_closed:
            try:
                snapshot=call_once('terminal_snapshot',budget.snapshot,diagnostic_dir=out/'diagnostics',db_path=budget.db_path)
                cumulative=cumulative_snapshot(old_stages,snapshot)
            except BaseException as secondary:snapshot_error=repr(secondary)
            try:call_once('close',budget.close,diagnostic_dir=out/'diagnostics',db_path=budget.db_path)
            except BaseException:pass
        terminal={'status':'technical_stop','error':repr(exc),'traceback':primary_trace,
            'sqlite_errorcode':getattr(exc,'sqlite_errorcode',None),'sqlite_errorname':getattr(exc,'sqlite_errorname',None),
            'segment_budget':snapshot,'cumulative_stages':cumulative,'snapshot_error':snapshot_error,
            'snapshot_from_final_close':ledger_closed,
            'budget_state_unresolved':snapshot is None,'automatic_retry':False}
        try:terminal['resources']=cumulative_resources(monitor,request)
        except BaseException as secondary:terminal['resources_error']=repr(secondary)
        try:write(out/'status.json',terminal)
        except BaseException as secondary:exc.add_note('Terminal status write failed: '+repr(secondary))
        raise

if __name__=='__main__':
    try:main()
    except Exception as exc:print(f'{type(exc).__name__}: {exc}');sys.exit(2)
