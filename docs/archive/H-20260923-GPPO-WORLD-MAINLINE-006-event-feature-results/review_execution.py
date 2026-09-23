"""Concentrated read-only review of completed H006; no model or env calls."""
from pathlib import Path
from collections import Counter
import hashlib,json,sqlite3,math
ROOT=Path(r"E:\Z博士\9.2日\WORLD-GPPO_9.11-replan-value-20260919-wt")
B=ROOT/'runs/finite-communication-ack-lease-fix-20260920'
P=B/'world-event-feature-execution-package-20260923'
RUN=B/'world-event-feature-authorized-run-20260923'
A=B/'world-event-feature-approval-20260923'
OUT=B/'world-event-feature-result-review-20260923'
OUT.mkdir(exist_ok=False)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rows(p):
 with Path(p).open(encoding='utf-8') as f:
  for line in f:
   if line.strip():yield json.loads(line)
def save(name,data):(OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
assert sha(P/'hashes.json')=='5ef0d32def14fa399e49ea38856b4504de69db48c7cd5f381b6820a576d9c8f3'
registry=read(P/'hashes.json')
assert all(sha(i['path'])==i['sha256'] for i in registry['files'])
analysis=read(RUN/'analysis.json');assert analysis['status']=='complete'
assert all(sha(p)==expected for p,expected in analysis['input_hashes'].items())
manifest=read(P/'paired-manifest.json');status=read(RUN/'run-status.json');cost=read(RUN/'runtime-costs.json')
assert len(manifest)==48 and len(analysis['paired_results'])==24
assert status['status']=='completed' and status['hard_counts']['env_steps']==558
for kind in ('optimizer_updates','world_updates','offline_updates'):assert status['hard_counts'][kind]==0
assert status['model_call_counts']['attempted']==status['model_call_counts']['completed']==dict(actor_readout=560,policy_encode=558,world_candidate_batch=558)
for name in ('authorization','runner','analyzer'):assert read(A/(name+'-command.json'))['exit_code']==0
assert not (RUN/'failure-ledger.jsonl').exists()
db=B/'ack-known-task-guard-baseline-v1/budget.sqlite3';before=sha(db)
with sqlite3.connect(db.as_uri()+'?mode=ro',uri=True) as c:
 c.row_factory=sqlite3.Row
 assert c.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
 stages=[dict(r) for r in c.execute('SELECT * FROM stages')]
 assert next(r for r in stages if r['stage']=='environment_steps')==dict(stage='environment_steps',limit_amount=1067,reserved=857,verified=857,unknown=0)
 for stage in stages:
  if stage['stage']!='environment_steps':assert stage['limit_amount']==stage['reserved']==stage['verified']==stage['unknown']==0
 reservations=[dict(r) for r in c.execute('SELECT * FROM reservations')]
 new=[r for r in reservations if r['run_id']=='world-event-feature-pair-20260923']
 old=[r for r in reservations if r['run_id']!='world-event-feature-pair-20260923']
 assert len(new)==558 and all(r['amount']==1 and r['status']=='verified' and r['attempt_id']=='world-event-feature-pair-20260923-attempt-0001' for r in new)
 assert sum(r['amount'] for r in old)==299
 with sqlite3.connect((A/'migration/pre-extension.sqlite3').as_uri()+'?mode=ro',uri=True) as historic:
  historic.row_factory=sqlite3.Row
  prior=[dict(r) for r in historic.execute('SELECT * FROM reservations')]
  assert sorted(old,key=lambda r:r['reservation_id'])==sorted(prior,key=lambda r:r['reservation_id'])
assert before==sha(db)
finals=list(rows(RUN/'budget-finalization.jsonl'))
assert len(finals)==558 and {r['reservation_id'] for r in finals}=={r['reservation_id'] for r in new}
features={}
for f in rows(RUN/'feature-ledger.jsonl'):
 key=(f['pair_id'],f['step']);arm=f['arm']
 assert arm not in features.setdefault(key,{})
 features[key][arm]=f
assert len(features)==279 and all(set(pair)=={'normal','event_features_off'} for pair in features.values())
same=[];stats=Counter()
for (pair_id,step),pair in features.items():
 n,o=pair['normal'],pair['event_features_off']
 fields=('public_observation_sha256','policy_hidden_before_sha256','world_hidden_before_sha256','legal_mask','candidate_features_raw_25x17','by_action_hidden_sha256','next_policy_hidden_sha256')
 equal=all(n[k]==o[k] for k in fields)
 stats['compared_steps']+=1;stats['identical_recorded_inputs']+=int(equal)
 if not equal:continue
 legal=[a for a in range(25) if n['legal_mask'][a]]
 ranks=lambda x:sorted(legal,key=lambda a:(-x['probabilities'][a],a))
 row=dict(pair_id=pair_id,step=step,probability_tv=sum(abs(x-y) for x,y in zip(n['probabilities'],o['probabilities']))/2,
  original_top1_changed=n['original_action']!=o['original_action'],selected_action_changed=n['selected_action']!=o['selected_action'],full_legal_rank_changed=ranks(n)!=ranks(o),
  normal_original=n['original_action'],off_original=o['original_action'],normal_selected=n['selected_action'],off_selected=o['selected_action'])
 same.append(row)
 for k in ('original_top1_changed','selected_action_changed','full_legal_rank_changed'):stats[k]+=int(row[k])
 stats['probabilities_changed']+=int(row['probability_tv']>0)
step_pairs={};feedback={'normal':Counter(),'event_features_off':Counter()};branch_initial=[]
for s in rows(RUN/'step-vector-rewards.jsonl'):
 key=(s['pair_id'],s['step']);compact={k:s[k] for k in ('action','submit_command','feedback','vector_reward','completed_before','completed_after','expired_before','expired_after','energy_before','energy_after','terminated','truncated')}
 assert s['arm'] not in step_pairs.setdefault(key,{})
 step_pairs[key][s['arm']]=compact;feedback[s['arm']][s['feedback']]+=1
assert len(step_pairs)==279 and all(set(v)=={'normal','event_features_off'} for v in step_pairs.values())
identical_steps=sum(v['normal']==v['event_features_off'] for v in step_pairs.values())
parents=[]
for i in range(8):
 parent=f'parent-{i:02d}';pairs=sorted([p for p in analysis['paired_results'] if p['parent_id']==parent],key=lambda p:p['repeat'])
 assert [p['repeat'] for p in pairs]==[0,1,2]
 diffs=[p['normal_minus_off_utility'] for p in pairs]
 parents.append(dict(parent_id=parent,normal_mean=sum(p['normal']['utility'] for p in pairs)/3,event_features_off_mean=sum(p['event_features_off']['utility'] for p in pairs)/3,repeat_differences=diffs,macro_difference=sum(diffs)/3,repeat_signs=[1 if d>0 else -1 if d<0 else 0 for d in diffs]))
assert all(p['repeat_differences']==[0,0,0] for p in parents)
utilities={arm:sum(p[arm+'_mean' if arm!='normal' else 'normal_mean'] for p in parents)/8 for arm in ('normal','event_features_off')}
review=dict(status='complete_review',identity_files_checked=len(registry['files']),analysis_input_hashes_checked=len(analysis['input_hashes']),branches=48,pairs=24,new_environment_steps=558,
 utilities=utilities,primary=analysis['primary'],parents=parents,secondary=analysis['secondary'],feedback_histograms={a:dict(v) for a,v in feedback.items()},
 recorded_input_comparison=dict(stats),same_input_probability_tv=dict(mean=sum(r['probability_tv'] for r in same)/len(same),max=max(r['probability_tv'] for r in same)),
 descriptive_trajectory_comparison=dict(steps_compared=279,identical_selected_action_reward_outcome_steps=identical_steps,note='post-run descriptive equality on actually recorded input/state hashes; no counterfactual execution or additional inference'),
 model_calls=status['model_call_counts'],timing=analysis['resource_evidence'],sqlite=dict(path=str(db),sha256=before,stages=stages,pending=0,unknown=0,integrity='ok',old_299_reservation_rows_unchanged=True,remaining_global=210,remaining_run_cap=210),
 review_resources=dict(env_step=0,model_forward=0,updates=0,new_attempt=0),technical_anomalies=['No dynamic technical stop recorded. Timing probe fields use different instrumentation boundaries; see timing note.'],
 timing_note='branch_timings.probe_seconds sums earlier timing_totals (1.4284s), whereas feature-ledger probe_seconds is captured later after tensor/JSON preparation (7.8780s). Non-probe remainder is environment+runner+some instrumentation, not isolated env.step cost. Same-input diagnostics add two actor evaluations; no claim of runtime saving.',
 decision='Retain current deployed/default features for compatibility; close this observed-development event-off sensitivity experiment as no observed incremental action/utility benefit. Do not infer equivalence or global ineffectiveness; no automatic training or followup experiment.',
 claims=['CI[0,0] is degenerate because all eight observed parent means are zero; frozen rule says inconclusive, not an equivalence bound.','Input suppression is not proved training-distribution neutral.','Observed development data/one model seed does not establish generalization.','H005 guard baseline contains frozen GPPO/world; whole-world independent value remains unresolved.'])
save('review.json',review);save('paired-input-comparison.json',same)
save('parent-results.json',parents)
save('sample-index.json',[dict(**row,actual_steps=next(p[row['arm']]['env_steps'] for p in analysis['paired_results'] if p['pair_id']==row['pair_id'])) for row in manifest])
print(json.dumps({k:review[k] for k in ('branches','pairs','new_environment_steps','utilities','recorded_input_comparison','same_input_probability_tv','feedback_histograms')},ensure_ascii=False))
