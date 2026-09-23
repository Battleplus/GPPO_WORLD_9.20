import importlib.util
import sqlite3
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('authorize_pair',Path(__file__).resolve().parents[1]/'tools/authorize_world_event_feature_pair_20260923.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def approval():
    return dict(status='approved',handoff_id='H-20260923-GPPO-WORLD-MAINLINE-006',package_hashes_sha256='package',approved_by='user',user_message_reference='test-only',approved_at='2026-09-23',accepted_event_sensitivity_only=True,old_limit=404,new_limit=1067,new_step_cap=768,model_call_caps=dict(policy_encode=768,world_candidate_batch=768,actor_readout=770),all_updates=0)

def database(tmp_path,pending=False):
    path=tmp_path/'synthetic.sqlite3'
    with sqlite3.connect(path) as c:
        c.execute('CREATE TABLE stages(stage TEXT PRIMARY KEY,limit_amount INTEGER,reserved INTEGER,verified INTEGER,unknown INTEGER)')
        c.executemany('INSERT INTO stages VALUES (?,?,?,?,?)',[('environment_steps',404,299,299,0),('optimizer_calls',0,0,0,0),('world_updates',0,0,0,0),('offline_updates',0,0,0,0)])
        c.execute('CREATE TABLE reservations(id TEXT,status TEXT)')
        c.execute('INSERT INTO reservations VALUES (?,?)',('old','pending' if pending else 'verified'))
        c.execute('CREATE TABLE metadata(name TEXT,value TEXT)')
        c.execute("INSERT INTO metadata VALUES ('history','unchanged')")
    return path

def test_approval_rejects_missing_or_expanded_scope():
    a=approval()
    m.validate_approval(a,'package')
    for key,value in [('status','pending'),('approved_at',''),('new_limit',1068),('all_updates',1),('package_hashes_sha256','other'),('accepted_event_sensitivity_only',False)]:
        with pytest.raises(ValueError):m.validate_approval({**a,key:value},'package')

def test_extension_changes_only_stage_limit_and_preserves_backup(tmp_path):
    path=database(tmp_path)
    with sqlite3.connect(path) as c:before=m.logical_rows(c)
    receipt=m.extend_same_database(path,m.sha(path),tmp_path/'backup.sqlite3',tmp_path/'receipt.json','approved-test')
    with sqlite3.connect(path) as c:after=m.logical_rows(c)
    before['stages']=[([r[0],1067,*r[2:]] if r[0]=='environment_steps' else r) for r in before['stages']]
    assert after==before and receipt['new_attempt']==0
    with sqlite3.connect(tmp_path/'backup.sqlite3') as c:
        assert c.execute("SELECT limit_amount FROM stages WHERE stage='environment_steps'").fetchone()==(404,)
    with pytest.raises(ValueError):m.extend_same_database(path,m.sha(path),tmp_path/'backup.sqlite3',tmp_path/'receipt.json','approved-test')

@pytest.mark.parametrize('fault',['pending','sha','state'])
def test_invalid_state_never_mutates_database(tmp_path,fault):
    path=database(tmp_path,pending=fault=='pending')
    if fault=='state':
        with sqlite3.connect(path) as c:c.execute("UPDATE stages SET reserved=300 WHERE stage='environment_steps'")
    before=m.sha(path)
    with pytest.raises(ValueError):m.extend_same_database(path,'wrong' if fault=='sha' else before,tmp_path/'backup.sqlite3',tmp_path/'receipt.json','test')
    assert m.sha(path)==before
