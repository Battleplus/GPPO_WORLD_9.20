"""One bound A -> B -> C runtime diagnostic; no training or retry."""
import time
ENTRY_WALL_STARTED=time.perf_counter()
import argparse
import math
import statistics
import struct
import traceback
from runtime_support import *
from attribution_contract import ATTEMPT, limits
from process_support import Monitor, OwnedWorker, Reservations
from session_budget import single_writer_budget
from sqlite_diagnostics import call_once
from analyze_runtime_necessity import analyze, percentile

COUNTERS=('environment_steps','environment_resets','model_loads','policy_encode',
          'world_candidate_batch','actor_readout','rule_decisions')

def verified_totals(snapshot,keys):
    stages=snapshot['stages']
    if any(v['pending'] or v['unknown'] for v in stages.values()):raise RuntimeError('Unresolved ledger')
    return {k:sum(v['verified'] for key,v in stages.items() if key.endswith('/'+k)) for k in keys}

def exact_delta(before,after,amounts):
    if set(before)!=set(COUNTERS) or set(after)!=set(COUNTERS):
        raise RuntimeError('Unexpected counter schema')
    if any(after[k]-before[k]!=amounts.get(k,0) for k in COUNTERS):
        raise RuntimeError('Actual operation counter mismatch')

def load_worker(stage,kind,auth,monitor,reservations):
    reservations.reserve({'model_loads':1})
    w=OwnedWorker(stage+'-'+kind,stage,kind,auth,monitor)
    exact_delta(dict.fromkeys(COUNTERS,0),w.ready['counts'],{'model_loads':1})
    w.last_counts=dict(w.ready['counts']);reservations.complete()
    return w

def checked_call(worker,command,amounts,reservations,**kwargs):
    reservations.reserve(amounts)
    result=worker.call(command,**kwargs)
    exact_delta(worker.last_counts,result['counts'],amounts)
    worker.last_counts=dict(result['counts'])
    return result

def semantic(records):
    return [{k:v for k,v in row.items() if k!='wall_seconds'} for row in records]

def replay_gate(auth,monitor,reservations):
    monitor.enter('A');folder=monitor.out/'A';folder.mkdir()
    workers={k:load_worker('A',k,auth,monitor,reservations) for k in ('full','reference','candidate')}
    batches={k:[] for k in workers};frozen={}
    for index in range(21):
        order=['full','reference','candidate']
        if index%2:order.reverse()
        current={}
        for kind in order:
            amounts={'policy_encode':25,'actor_readout':25,'world_candidate_batch':25 if kind=='full' else 0}
            result=checked_call(workers[kind],'replay_once',amounts,reservations)
            if result['stats']['decisions']!=25 or len(result['records'])!=25:
                raise RuntimeError('Replay decision count mismatch')
            if any(not math.isfinite(result['stats'][k]) or result['stats'][k]<0 for k in ('cpu_seconds','wall_seconds')):
                raise RuntimeError('Invalid replay timing')
            write(folder/f'{index:02d}-{kind}.json',result);reservations.complete()
            values=semantic(result['records'])
            if kind in frozen and values!=frozen[kind]:raise RuntimeError('Replay changed across batches')
            frozen[kind]=values;current[kind]=values
            batches[kind].append({**result['stats'],'index':index,'cold':index==0,
                'decision_wall_seconds':[r['wall_seconds'] for r in result['records']]})
        if current['reference']!=current['candidate']:raise RuntimeError('Neutralized-native versus optimized N mismatch')
        monitor.check(disk=True)
    for w in workers.values():w.close()
    cpu={k:sum(b['cpu_seconds'] for b in bs) for k,bs in batches.items()}
    if min(cpu.values())<=0:raise RuntimeError('Unmeasurable replay CPU')
    mean=cpu['candidate']/525*1000
    wall=percentile([v*1000 for b in batches['candidate'] for v in b['decision_wall_seconds']],.95)
    result={'batches':batches,'bitwise_reference_candidate_match':True,'full_historical_match':True,
        'paired_replay_cpu_ratio':cpu['candidate']/cpu['full'],
        'N_mean_cpu_ms':mean,'N_wall_p95_ms':wall,'all_21_batches_primary':True,
        'first_batch_ratios':{'N_over_M':batches['candidate'][0]['cpu_seconds']/batches['full'][0]['cpu_seconds'] if batches['full'][0]['cpu_seconds'] else None},
        'warm_20_ratio':sum(b['cpu_seconds'] for b in batches['candidate'][1:])/sum(b['cpu_seconds'] for b in batches['full'][1:]),
        'resources':monitor.snapshot()}
    write(folder/'result.json',result)
    if mean>10 or wall>50:raise RuntimeError('A absolute controller cost gate')
    monitor.check(disk=True);write(folder/'status.json',{'status':'passed'})
    return result

def validate_matrix(rows,baseline,tapes):
    from prepare_minimal import selected_keys
    expected=[(p,r,'N') for p,r in selected_keys()]
    if [(r['parent'],r['repeat'],r['arm']) for r in rows]!=expected:raise RuntimeError('Fixed selected matrix mismatch')
    old={(r['parent'],r['repeat'],r['arm']):r for r in baseline}
    original={(f'eval-{p:04d}',r,a) for p in range(1600) for r in range(3) for a in ('M','R')}
    if len(old)!=9600 or len(baseline)!=9600 or set(old)!=original:raise RuntimeError('Historical matrix mismatch')
    if len(tapes['tapes'])!=1600 or len(tapes['hashes'])!=1600:raise RuntimeError('Tape count mismatch')
    if [digest(t) for t in tapes['tapes']]!=tapes['hashes']:raise RuntimeError('Tape identity mismatch')
    for row in rows:
        p,r=row['parent'],row['repeat'];index=int(p.split('-')[1])
        if row['tape_index']!=index or row['tape_hash']!=tapes['hashes'][index]:raise RuntimeError('Wrong tape mapping')
        if row['exogenous_key']!=f'frozen-native-public-history-allocation-v1|{p}|repeat-{r}':raise RuntimeError('Wrong exogenous key')
        if not row['initial_public_hash']==old[(p,r,'M')]['initial_public_hash']==old[(p,r,'R')]['initial_public_hash']:raise RuntimeError('Historical pairing mismatch')

def validate_step(row,step,memory,previous,energy):
    if row['step']!=step or row['submit_command'] is not True:raise RuntimeError('Step contract mismatch')
    public=row['public'];memory.observe(public);candidates=list(memory.candidates(public))
    if row['candidates']!=candidates or row['action'] not in candidates or not public['mask'][row['action']]:
        raise RuntimeError('Public guard or mask mismatch')
    memory.submitted(public,row['action']);info=row['info']
    f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
    en=sum(info['energy'].values());counts=info['counts']
    vec=[f32((max(0,counts['completed']-previous['completed'])-max(0,counts['expired']-previous['expired']))/6),f32(-max(0,energy-en)/36)]
    if vec!=row['vector_reward'] or not all(math.isfinite(v) for v in vec):raise RuntimeError('Reward recomputation mismatch')
    if row['diagnostic']['world_enabled'] is not False or 'selected_world_hidden_sha256' in row['diagnostic']:
        raise RuntimeError('Inactive world represented as active')
    if any(not math.isfinite(row['controller'][k]) or row['controller'][k]<0 for k in ('cpu','wall')):raise RuntimeError('Invalid controller cost')
    ending(row['done'],info)
    return counts,en,.99**step*(.4*vec[0]+.2*vec[1])

def publish_progress(monitor,stage,completed,key):
    try:
        with (monitor.out/'observer-progress.jsonl').open('a',encoding='utf-8') as f:
            f.write(canonical({'stage':stage,'completed':completed,'last_key':key})+'\n');f.flush()
    except OSError:
        monitor.observer_publish_errors=getattr(monitor,'observer_publish_errors',0)+1

def environments(stage,auth,monitor,reservations,inputs):
    monitor.enter(stage);folder=monitor.out/stage;folder.mkdir()
    w=load_worker(stage,'candidate',auth,monitor,reservations)
    if stage=='B':
        tapes=w.call('generate',count=2,seed=925310000)
        formal_hashes=set(read(inputs['scenario_manifest'])['hashes'])
        if len(tapes['tapes'])!=2 or len(set(tapes['hashes']))!=2 or formal_hashes.intersection(tapes['hashes']):raise RuntimeError('Technical tape overlap')
        rows=[{'parent':f'technical-{p:04d}','repeat':0,'arm':'N','tape_index':p,'tape_hash':h,
            'exogenous_key':f'{ATTEMPT}|technical-{p:04d}|repeat-0'} for p,h in enumerate(tapes['hashes'])]
    else:
        tapes=read(inputs['scenario_manifest']);rows=read(ROOT/'sample-manifest.json')
        validate_matrix(rows,read(inputs['baseline_episodes']),tapes)
    write(folder/'scenario-manifest.json',tapes)
    logger=CompressedRecords(folder/'steps.jsonl.gz');episodes=[];latencies=[];parts={}
    try:
        for item in rows:
            parent,repeat=item['parent'],item['repeat'];key=(parent,repeat,'N')
            reset=checked_call(w,'reset',{'environment_resets':1},reservations,tape=tapes['tapes'][item['tape_index']],key=item['exogenous_key'])
            if digest(reset['initial_public'])!=reset['initial_hash']:raise RuntimeError('Reset public digest mismatch')
            if stage=='C' and reset['initial_hash']!=item['initial_public_hash']:raise RuntimeError('Native reset not paired with M/R')
            append(folder/'episode-starts.jsonl',{'key':key,'exogenous_key':item['exogenous_key'],'reset':reset});reservations.complete()
            memory=PublicMemory();previous={'completed':0,'expired':0};energy=36.;utility=0.
            for step in range(18):
                row=checked_call(w,'step',{'environment_steps':1,'policy_encode':1,'actor_readout':1},reservations)
                previous,energy,value=validate_step(row,step,memory,previous,energy);utility+=value
                if step==0 and digest(row['public'])!=reset['initial_hash']:raise RuntimeError('First public state differs from reset')
                row.update(parent=parent,repeat=repeat,arm='N',reservations=list(reservations.pending))
                monitor.measure('logging',logger.add,row);reservations.complete();latencies.append(row['controller'])
                for tag in ('controller','environment','labels'):
                    part=parts.setdefault(tag,{'cpu':0.,'wall':0.})
                    for k,v in row[tag].items():part[k]+=v
                if row['done']:
                    ep=row['episode']
                    if ep['steps']!=step+1 or abs(ep['utility']-utility)>1e-12 or abs(ep['energy_used']-(36-energy))>1e-12:
                        raise RuntimeError('Episode summary mismatch')
                    records=row['info']['completion_records']
                    for field,label in [('physical_on_time','physical_arrival_before_deadline'),('host_on_time_observed','host_confirmation_before_deadline')]:
                        if ep[field]!=sum(v[label] is True for v in records.values())/6:raise RuntimeError('Completion label mismatch')
                    ep.update(parent=parent,repeat=repeat,arm='N',initial_public_hash=reset['initial_hash'])
                    append(folder/'episodes.jsonl',ep);episodes.append(ep);publish_progress(monitor,stage,len(episodes),key);break
            else:raise RuntimeError('Episode failed native termination within18')
            monitor.check(disk=True)
        logger.close();verify_compressed(folder/'steps.jsonl.gz',logger.digest.hexdigest(),logger.rows)
        w.close()
        if len(episodes)!=len(rows):raise RuntimeError('Incomplete stage')
        runtime={'latencies':latencies,'worker_parts':parts,'compressed_digest':logger.digest.hexdigest(),
            'compressed_rows':logger.rows,'resources':monitor.snapshot()}
        write(folder/'cost-and-interface.json',runtime);monitor.check(disk=True)
        write(folder/'status.json',{'status':'passed','episodes':len(episodes),'updates':0})
        return episodes,runtime
    finally:
        if not logger.raw.closed:logger.close()

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--authorization',type=Path);args=p.parse_args(argv)
    authorization,manifest,request=authorize(args.authorization)
    if os.name!='nt':raise RuntimeError('Pinned Windows runtime required')
    out=ROOT/'runs/runtime-necessity-v1';out.mkdir(parents=True,exist_ok=False)
    write(out/'authorization.json',authorization);write(out/'execution-manifest.json',manifest)
    monitor=Monitor(request,out);monitor.started=ENTRY_WALL_STARTED;monitor.stage_start=ENTRY_WALL_STARTED
    monitor.enter('A');budget=None;reservations=None;closed=False;snapshot=None
    try:
        native=read(PREP/'binding.json');caps=limits(request)
        Budget=single_writer_budget(budget_class(native))
        budget=call_once('initialize',Budget,out/'budget.sqlite3',limits=caps,run_limits=caps,attempt_id=ATTEMPT,
            run_id=ATTEMPT,diagnostic_dir=out/'diagnostics',db_path=out/'budget.sqlite3')
        reservations=Reservations(budget,monitor)
        a=replay_gate(args.authorization,monitor,reservations)
        environments('B',args.authorization,monitor,reservations,request['inputs'])
        episodes,cost=environments('C',args.authorization,monitor,reservations,request['inputs'])
        selection=read(ROOT/'sample-manifest.json');selected={(r['parent'],r['repeat']) for r in selection}
        old=[r for r in read(request['inputs']['baseline_episodes']) if (r['parent'],r['repeat']) in selected]
        historical_cost=read(ROOT/'selected-historical-M-cost.json')
        resources=monitor.snapshot()
        result=analyze(old,episodes,cost,a,resources,selection,historical_cost,check=lambda:monitor.check())
        write(out/'C/analysis.json',result);write(out/'C/complete-N-episodes.json',episodes)
        snapshot=call_once('snapshot',budget.snapshot,diagnostic_dir=out/'diagnostics',db_path=budget.db_path)
        new_totals=verified_totals(snapshot,request['history_counters'])
        integrity=budget._session.execute('PRAGMA integrity_check').fetchone()[0]
        if integrity!='ok':raise RuntimeError('SQLite integrity failed')
        write(out/'budget-snapshot.json',snapshot)
        call_once('close',budget.close,diagnostic_dir=out/'diagnostics',db_path=budget.db_path);closed=True
        authorize(args.authorization);monitor.check(disk=True)
        history=read(request['inputs']['historical_verification'])
        status={'status':'completed','decision':result['decision'],'new_budget':{k:v for k,v in snapshot.items() if k!='history'},
            'full_budget_snapshot_sha256':sha(out/'budget-snapshot.json'),'sqlite_integrity':integrity,
            'new_totals':new_totals,'stage2_historical_totals':history['cumulative_A_B_C'],
            'stage2_plus_this_diagnostic_totals':{k:history['cumulative_A_B_C'][k]+v for k,v in new_totals.items()},
            'historical_resources':history['resources'],'new_resources':monitor.snapshot(),
            'resource_scope':'stage2 plus this diagnostic only; earlier research not reset or refunded',
            'updates':0,'automatic_retry':False,'further_experiments_authorized':False,
            'observer_publish_errors':getattr(monitor,'observer_publish_errors',0)}
        report=f"# Frozen runtime diagnostic\n\nDecision: `{result['decision']}`.\n\n800 new N episodes; 1600 corresponding historical M/R episodes reused. Stage2 full9600 conclusion unchanged.\n\nUtility contrasts: {canonical(result['summary'])}\n\nCosts: {canonical(result['N_controller_cost'])}\n\nWorld-independent contribution is not established by this removal test. No automatic training or follow-up. Full process costs and historical lineage are in status.json.\n"
        (out/'report.md').write_text(report,encoding='utf-8');monitor.check(disk=True)
        status['new_resources']=monitor.snapshot()
        status['final_parent_tail_bound']=request['recovery_shutdown_holdback']
        status['tail_is_reserved_upper_bound_not_exact_measurement']=True
        write(out/'status.json',status)
    except BaseException as exc:
        trace=traceback.format_exc()
        if reservations is not None:reservations.unknown(exc)
        for child in monitor.children:
            try:child.kill()
            except Exception as secondary:exc.add_note('Worker cleanup: '+repr(secondary))
        if budget is not None and not closed:
            try:snapshot=call_once('terminal_snapshot',budget.snapshot,diagnostic_dir=out/'diagnostics',db_path=budget.db_path)
            except Exception as secondary:exc.add_note('Snapshot: '+repr(secondary))
            try:call_once('close',budget.close,diagnostic_dir=out/'diagnostics',db_path=budget.db_path)
            except Exception as secondary:exc.add_note('Ledger close: '+repr(secondary))
        terminal={'status':'technical_stop','error':repr(exc),'traceback':trace,'new_budget':snapshot,'automatic_retry':False}
        try:terminal['resources']=monitor.snapshot()
        except Exception as secondary:terminal['resource_error']=repr(secondary)
        try:write(out/'technical-stop.json',terminal)
        except Exception as secondary:exc.add_note('Terminal write: '+repr(secondary))
        raise

if __name__=='__main__':main()
