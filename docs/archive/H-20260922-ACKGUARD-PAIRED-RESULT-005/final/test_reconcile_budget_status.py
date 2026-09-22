from copy import deepcopy
import pytest
from reconcile_budget_status import reconcile

def sample():
    s=[dict(reservation_id="r",branch_id="b",step=1,budget_status="reserved_pending_finalization",vector_reward=[1,-.1],action=2)]
    f=[dict(reservation_id="r",branch_id="b",step=1,status="verified",actual_reservation_status="verified")]
    d=[dict(reservation_id="r",status="verified",amount=1,stage="environment_steps",run_id="run",attempt_id="attempt")]
    return s,f,d

def test_matching_preserves_raw_and_rewards():
    s,f,d=sample();before=deepcopy(s)
    result=reconcile(s,f,d,"run","attempt")
    assert s==before and result[0]["budget_status"]=="verified"
    assert result[0]["vector_reward"]==s[0]["vector_reward"] and result[0]["original_budget_status"]==s[0]["budget_status"]

@pytest.mark.parametrize("bad",["missing","duplicate","wrong_branch","wrong_step","pending","db_unknown","wrong_run","wrong_amount"])
def test_rejects_unproved_finalization(bad):
    s,f,d=sample()
    if bad=="missing":f=[]
    elif bad=="duplicate":f+=deepcopy(f)
    elif bad=="wrong_branch":f[0]["branch_id"]="other"
    elif bad=="wrong_step":f[0]["step"]=2
    elif bad=="pending":f[0]["status"]="pending"
    elif bad=="db_unknown":d[0]["status"]="unknown"
    elif bad=="wrong_run":d[0]["run_id"]="other"
    elif bad=="wrong_amount":d[0]["amount"]=2
    with pytest.raises(ValueError):reconcile(s,f,d,"run","attempt")
