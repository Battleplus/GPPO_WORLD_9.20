from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "run_world_event_feature_pair_20260923.py"
spec = importlib.util.spec_from_file_location("world_event_feature_pair", SCRIPT)
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def _copy_data(value):
    return copy.deepcopy(value)


def _shape(value):
    if not isinstance(value, list):
        return ()
    return (len(value),) + (_shape(value[0]) if value else ())


class _FakeTensor:
    def __init__(self, data):
        self.data = _copy_data(data)

    @property
    def shape(self):
        return _shape(self.data)

    def clone(self):
        return _FakeTensor(self.data)

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return _copy_data(self.data)

    def item(self):
        value = self.data
        while isinstance(value, list):
            value = value[0]
        return value

    def reshape(self, *shape):
        requested = list(shape)
        flat = []

        def flatten(value):
            if isinstance(value, list):
                for child in value:
                    flatten(child)
            else:
                flat.append(value)

        flatten(self.data)
        if requested.count(-1) == 1:
            known = math.prod(value for value in requested if value != -1)
            requested[requested.index(-1)] = len(flat) // known
        assert math.prod(requested) == len(flat)
        cursor = 0

        def build(dim):
            nonlocal cursor
            if dim == len(requested):
                value = flat[cursor]
                cursor += 1
                return value
            return [build(dim + 1) for _ in range(requested[dim])]

        return _FakeTensor(build(0))

    def squeeze(self, dim=-1):
        dims = self.shape
        dim = len(dims) + dim if dim < 0 else dim
        assert dims[dim] == 1

        def remove(value, depth):
            if depth == dim:
                return remove(value[0], depth + 1)
            if not isinstance(value, list):
                return value
            return [remove(child, depth + 1) for child in value]

        return _FakeTensor(remove(self.data, 0))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        indices = index if isinstance(index, tuple) else (index,)

        def select(value, remaining):
            if not remaining:
                return value
            head, *tail = remaining
            if head is None:
                return [select(value, tuple(tail))]
            if isinstance(head, slice):
                return [select(child, tuple(tail)) for child in value[head]]
            return select(value[head], tuple(tail))

        return _FakeTensor(select(self.data, indices))

    def __setitem__(self, index, value):
        indices = index if isinstance(index, tuple) else (index,)
        replacement = value.data if isinstance(value, _FakeTensor) else value

        def assign(target, remaining):
            head, *tail = remaining
            if isinstance(head, slice):
                for offset in range(*head.indices(len(target))):
                    if tail:
                        assign(target[offset], tuple(tail))
                    else:
                        target[offset] = _copy_data(replacement)
                return
            if tail:
                assign(target[head], tuple(tail))
            else:
                target[head] = _copy_data(replacement)

        assign(self.data, indices)


class _FakeNoGrad:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class _FakeTorch:
    float32 = object()
    bool = object()

    @staticmethod
    def zeros(shape, **_kwargs):
        def build(dim):
            if dim == len(shape):
                return 0.0
            return [build(dim + 1) for _ in range(shape[dim])]

        return _FakeTensor(build(0))

    @staticmethod
    def as_tensor(value, **_kwargs):
        return _FakeTensor(value)

    @staticmethod
    def cat(values, dim=-1):
        assert dim in (-1, 1)
        left, right = (value.data for value in values)
        return _FakeTensor([left_row + right_row for left_row, right_row in zip(left, right)])

    @staticmethod
    def argmax(value, dim=-1):
        assert dim in (-1, 1)
        rows = value.data
        return _FakeTensor([max(range(len(row)), key=lambda index: row[index]) for row in rows])

    @staticmethod
    def no_grad():
        return _FakeNoGrad()


class _FakeHook:
    def __init__(self, hooks, callback):
        self._hooks = hooks
        self._callback = callback

    def remove(self):
        if self._callback in self._hooks:
            self._hooks.remove(self._callback)


class _FakeModule:
    def __init__(self, function):
        self._function = function
        self._hooks = []

    def register_forward_hook(self, callback):
        self._hooks.append(callback)
        return _FakeHook(self._hooks, callback)

    def __call__(self, *args):
        output = self._function(*args)
        for callback in list(self._hooks):
            callback(self, args, output)
        return output


class _FakePolicy:
    def __init__(self):
        self.base = type("Base", (), {})()
        self.base.pair_actor = _FakeModule(lambda _features: _FakeTensor([[0.0] * 24]))
        self.base.noop_actor = _FakeModule(lambda _features: _FakeTensor([[0.0]]))
        self.preference_actor = _FakeModule(lambda _features: _FakeTensor([[0.1] * 25]))
        self.candidate_actor = _FakeModule(
            lambda candidates: _FakeTensor([[[sum(float(value) for value in row[12:17])] for row in candidates.data[0]]])
        )

    def encode(self, _obs, _hidden):
        return _FakeTensor([[0.0]]), _FakeTensor([[0.0]]), _FakeTensor([[[0.0] * 128]])

    def evaluate_encoded(self, features, _pair_messages, _preference, actor_candidates, _mask):
        pair = self.base.pair_actor(features)
        noop = self.base.noop_actor(features)
        preference = self.preference_actor(features)
        candidate = self.candidate_actor(actor_candidates).squeeze(-1)
        base = _FakeTorch.cat((pair.reshape(1, -1), noop.reshape(1, -1)), dim=-1).data[0]
        candidate_values = candidate.data[0]
        preference_values = preference.data[0]
        logits = [base[index] + preference_values[index] + candidate_values[index] for index in range(25)]
        maximum=max(logits)
        weights = [math.exp(value-maximum) if _mask.data[0][i] else 0.0 for i,value in enumerate(logits)]
        total = sum(weights)
        probabilities = [value / total for value in weights]
        distribution = type("Distribution", (), {"probs": _FakeTensor([probabilities])})()
        return {"logits": _FakeTensor([logits]), "distribution": distribution}


class _FakeWorld:
    def predict_all_candidates(self, _features, _hidden, _obs, use_events=True):
        assert use_events is True
        rows = [[float(action + column + 1) for column in range(17)] for action in range(25)]
        by_action = {action: {"hidden": _FakeTensor([float(action)])} for action in range(25)}
        return _FakeTensor([rows]), by_action


def _fake_runtime():
    classes = [None] * 6
    classes[4] = lambda mask: list(mask)
    classes[5] = lambda obs, _device: _FakeTensor([[float(value) for value in obs.get("flat", [0.0])]])
    return {
        "classes": classes,
        "policy": _FakePolicy(),
        "world": _FakeWorld(),
        "preference": None,
        "device": "cpu",
    }


def _feature(arm: str, *, pair: str = runner.FIRST_PAIR_ID, hidden: str = "h", off: bool = False, step: int = 1) -> dict:
    raw = [[float(action + column) for column in range(17)] for action in range(25)]
    actor = [row[:12] + ([0.0] * 5 if off else row[12:]) for row in raw]
    return {
        "schema": runner.FEATURE_SCHEMA,
        "branch_id": f"{pair}|{arm}",
        "pair_id": pair,
        "arm": arm,
        "exogenous_key": "exo-p",
        "step": step,
        "public_observation_sha256": "obs",
        "policy_hidden_before_sha256": hidden,
        "world_hidden_before_sha256": "world",
        "candidate_features_raw_25x17": raw,
        "candidate_features_actor_25x17": actor,
        "event_features_before_25x5": [row[12:] for row in raw],
        "event_features_after_25x5": [row[12:] if not off else [0.0] * 5 for row in raw],
        "by_action_hidden_sha256": {str(action): f"by-{action}" for action in range(25)},
        "next_policy_hidden_sha256": "next",
        "base_logits": [0.0] * 25,
        "preference_logits": [0.0] * 25,
        "candidate_logits": [0.0] * 25,
        "logits": [0.0] * 25,
        "probabilities": [1.0 / 25.0] * 25,
        "timing": {},
        "model_calls": {"policy_encode": 1, "world_candidate_batch": 1, "actor_readout": 1},
        "model_call_status": {
            "policy_encode": {"attempted": 1, "completed": 1},
            "world_candidate_batch": {"attempted": 1, "completed": 1},
            "actor_readout": {"attempted": 1, "completed": 1},
        },
    }


def _write_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


class _FakeCounters:
    _NAMES = (
        "probe_calls", "env_step_calls", "env_steps", "verified_steps", "model_forward", "actor_forward",
        "world_forward", "optimizer_updates", "world_updates", "offline_updates", "branches_completed",
    )

    def __init__(self):
        for name in self._NAMES:
            setattr(self, name, 0)

    def as_dict(self):
        values = {name: int(getattr(self, name)) for name in self._NAMES}
        values["successful_env_steps"] = values["env_steps"]
        return values


class _FakeBudget:
    def __init__(self, *_args, **_kwargs):
        self.steps = 0

    def run_totals(self, _run_id):
        return {
            "environment_steps": {
                "reserved": self.steps,
                "verified": self.steps,
                "unknown": 0,
                "pending": 0,
            }
        }

    def export_snapshot(self, path):
        path.write_text(json.dumps({"steps": self.steps}) + "\n", encoding="utf-8")


class _FakeRuntimeAdapters:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeBaseAdapters:
    def __init__(self):
        self.close = None

    def load_snapshots(self, prefix_ids):
        return {prefix_id: {"prefix_id": prefix_id} for prefix_id in prefix_ids}

    def load_runtime(self, _snapshots):
        return _fake_runtime()

    def runtime_digest(self, _runtime):
        return "frozen-runtime"

    def reward(self, *_args):
        raise AssertionError("fake branch must not call the environment reward adapter")

    def task_capacity(self, _env):
        return 2


def _stub_h005(branch_calls, *, fail_at=None):
    stable = type("FakeH005", (), {})()
    stable.PersistentBudget = _FakeBudget
    stable.RuntimeAdapters = _FakeRuntimeAdapters
    stable.Counters = _FakeCounters
    stable.observation_digest = runner._stable_observation_digest
    stable.hidden_digest = runner._stable_hidden_digest
    base = _FakeBaseAdapters()
    stable._default_adapters = lambda: base
    stable.MAX_TOTAL_STEPS = 0
    stable.MAX_STEPS_PER_BRANCH = 0

    def execute_branch(item, _snapshot, runtime, adapters, budget, counters, staging):
        order = int(item["order"])
        branch_calls.append(order)
        obs = {
            "flat": [1.0, 2.0],
            "mask": [True] * 25,
            "version": 1,
            "time": 0,
            "public_entity_ids": ["entity-0"],
            "continuation_actions": [],
            "trigger_flags": {},
        }
        probe = adapters.probe(runtime, obs, None, None)
        counters.probe_calls += 1
        counters.model_forward += 1
        counters.actor_forward += 1
        counters.world_forward += 1
        counters.env_step_calls += 1
        counters.env_steps += 1
        counters.verified_steps += 1
        budget.steps += 1
        action = int(probe["original_action"])
        _write_jsonl(staging / "decision-ledger.jsonl", {
            "step": 1,
            "original_action": action,
            "final_action": action,
            "candidate_actions": [action],
            "excluded_actions": [],
        })
        _write_jsonl(staging / "step-vector-rewards.jsonl", {"step": 1, "vector_reward": [0.0, 0.0]})
        _write_jsonl(staging / "budget-finalization.jsonl", {"step": 1, "reservation_id": f"reservation-{order}"})
        if fail_at is not None and order == fail_at:
            _write_jsonl(staging / "failure-ledger.jsonl", {"branch_id": item["branch_id"], "step": 1, "reason": "stub failure"})
            raise RuntimeError(f"stub branch failure at order {order}")
        return {"branch_id": item["branch_id"], "step_count": 1, "guard_trigger_count": 0}

    stable.execute_branch = execute_branch
    return stable


def _run_stubbed_execute(monkeypatch, output_dir: Path, *, fail_at=None):
    branch_calls = []
    stable = _stub_h005(branch_calls, fail_at=fail_at)
    monkeypatch.setattr(runner, "validate_authorization", lambda *_args, **_kwargs: {
        "authorization": {"run": {"run_id": "stub-run", "attempt_id": "stub-attempt"}},
        "budget": {},
    })
    monkeypatch.setattr(runner, "_load_h005_runner_from_path", lambda: stable)
    original_init = runner.FeatureCapture.__init__
    init_calls = []

    def capture_init(self, item, arm, *args, **kwargs):
        init_calls.append({"order": int(item["order"]), "diagnostic_first_pair": kwargs.get("diagnostic_first_pair"), "observation_digest_fn": kwargs.get("observation_digest_fn"), "hidden_digest_fn": kwargs.get("hidden_digest_fn")})
        kwargs["torch_module"] = _FakeTorch()
        return original_init(self, item, arm, *args, **kwargs)

    monkeypatch.setattr(runner.FeatureCapture, "__init__", capture_init)
    auth_path = output_dir.parent / "stub-auth.json"
    auth_path.write_text("{}\n", encoding="utf-8")
    return runner.execute_authorized(auth_path, manifest_path=runner.MANIFEST, output_dir=output_dir), branch_calls, init_calls


def _template(tmp_path: Path) -> Path:
    payload = runner.authorization_template(runner.package_validation())
    payload["status"] = "authorized"
    payload["review"] = {"reviewed_by": "test", "reviewed_at": "2026-09-23T00:00:00Z", "accepted_interpretation": True}
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_static_package_has_fixed_pairs_and_native_source_pin():
    result = runner.package_validation()
    assert result["ok"] is True
    assert result["manifest"]["row_count"] == 48
    assert result["manifest"]["pair_count"] == 24
    assert result["native_source_manifest"]["ok"] is True
    assert result["native_source_manifest"]["file_count"] == 44
    assert result["budget_readonly"]["stages"]["environment_steps"]["limit"] == 404
    assert result["hard_counts"]["env_step"] == 0


def test_manifest_rejects_duplicate_or_wrong_order():
    rows = runner._manifest_rows(runner.MANIFEST)
    broken = [dict(row) for row in rows]
    broken[1]["arm"] = "normal"
    result = runner.validate_manifest(broken)
    assert result["ok"] is False
    assert any("one normal/one event_features_off" in reason for reason in result["failures"])

    broken = [dict(row) for row in rows]
    broken[0]["order"], broken[1]["order"] = broken[1]["order"], broken[0]["order"]
    result = runner.validate_manifest(broken)
    assert result["ok"] is False
    assert any("order" in reason for reason in result["failures"])


def test_first_pair_gate_requires_two_arms_and_preserves_non_event_columns():
    normal = _feature("normal")
    off = _feature("event_features_off", off=True)
    gate = runner.compare_first_pair([normal, off])
    assert gate["ok"] is True
    assert gate["pair_id"] == runner.FIRST_PAIR_ID
    assert gate["step"] == 1
    assert gate["checks"]["same_raw_candidate_features"] is True
    assert gate["checks"]["same_by_action_hidden"] is True
    assert gate["checks"]["same_candidate_columns_0_12"] is True
    assert runner.compare_first_pair([normal, _feature("normal", off=True)])["ok"] is False

    broken = _feature("event_features_off", off=True, hidden="different")
    assert runner.compare_first_pair([normal, broken])["ok"] is False


@pytest.mark.parametrize(
    "mutate",
    [
        lambda rows: rows.pop(0),
        lambda rows: rows.append(_feature("normal")),
        lambda rows: rows.__setitem__(0, {**rows[0], "arm": "unknown"}),
        lambda rows: rows[0].pop("candidate_logits"),
    ],
)
def test_first_pair_gate_rejects_missing_duplicate_unknown_or_incomplete_step_one(mutate):
    rows = [_feature("normal"), _feature("event_features_off", off=True)]
    mutate(rows)
    assert runner.compare_first_pair(rows)["ok"] is False


def test_first_pair_gate_ignores_later_steps_and_other_pairs():
    normal = _feature("normal")
    off = _feature("event_features_off", off=True)
    later = _feature("normal", step=2)
    later["arm"] = "fork-record"
    later["candidate_logits"] = None
    other = _feature("normal", pair="parent-07|W1|seed-1101|prefix-0|repeat-2")
    assert runner.compare_first_pair([later, other, normal, off])["ok"] is True
    assert runner.compare_first_pair([other])["ok"] is False


def test_feature_schema_detects_missing_required_logs():
    row = _feature("normal")
    assert runner.required_feature_fields(row) == []
    del row["candidate_logits"]
    assert runner.required_feature_fields(row) == ["candidate_logits"]


def test_authorization_missing_and_wrong_source_hash_fail_closed(tmp_path: Path):
    with pytest.raises(runner.AuthorizationError, match="explicit reviewed authorization"):
        runner.validate_authorization(tmp_path / "missing.json", output_dir=tmp_path / "out")

    path = _template(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["source"]["runner_sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(runner.AuthorizationError, match="source hash mismatch"):
        runner.validate_authorization(path, output_dir=tmp_path / "out")


def test_authorization_wrong_budget_identity_and_limit_fail_closed(tmp_path: Path):
    path = _template(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["budget"]["sha256"] = "1" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(runner.AuthorizationError, match="budget SQLite hash mismatch"):
        runner.validate_authorization(path, output_dir=tmp_path / "out")


def test_model_call_sidecar_separates_attempts_from_completed_calls():
    counts = runner.ModelCallCounters()
    counts.start("world_candidate_batch")
    assert counts.as_dict()["attempted"]["world_candidate_batch"] == 1
    assert counts.as_dict()["completed"]["world_candidate_batch"] == 0
    counts.finish("world_candidate_batch")
    assert counts.as_dict()["completed"]["world_candidate_batch"] == 1


def test_execute_authorized_stub_runs_all_manifest_rows_and_real_probe(monkeypatch):
    with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
        output = Path(temp_dir) / "successful-run"
        status, branch_calls, init_calls = _run_stubbed_execute(monkeypatch, output)

        assert status["status"] == "completed"
        assert branch_calls == list(range(1, 49))
        assert len(init_calls) == 48
        assert [call["order"] for call in init_calls[:2]] == [1, 2]
        assert all(call["diagnostic_first_pair"] is True for call in init_calls[:2])
        assert all(call["diagnostic_first_pair"] is False for call in init_calls[2:])
        assert all(call["observation_digest_fn"] is runner._stable_observation_digest for call in init_calls)
        assert all(call["hidden_digest_fn"] is runner._stable_hidden_digest for call in init_calls)

        gate = json.loads((output / "first-pair-gate.json").read_text(encoding="utf-8"))
        assert gate["ok"] is True
        assert gate["pair_id"] == runner.FIRST_PAIR_ID
        assert gate["step"] == 1
        features = [json.loads(line) for line in (output / "feature-ledger.jsonl").read_text(encoding="utf-8").splitlines()]
        first = [row for row in features if row["pair_id"] == runner.FIRST_PAIR_ID]
        assert len(first) == 2
        assert all(row["pair_diagnostic"]["enabled"] is True for row in first)
        assert all(row["pair_diagnostic"]["extra_actor_readout"] is not None for row in first)
        assert {row["model_calls"]["actor_readout"] for row in first} == {2}
        assert all(row["model_calls"]["actor_readout"] == 1 for row in features[2:])
        assert json.loads((output / "runtime-costs.json").read_text(encoding="utf-8"))["status"] == "completed"
        assert json.loads((output / "run-status.json").read_text(encoding="utf-8"))["status"] == "completed"
        # Exercise the real analyzer against records produced by the real probe;
        # only missing historical step evidence is supplied by this stub.
        analysis_spec=importlib.util.spec_from_file_location('stub_output_analyzer',ROOT/'tools/analyze_world_event_feature_pair_20260923.py')
        analysis=importlib.util.module_from_spec(analysis_spec)
        analysis_spec.loader.exec_module(analysis)
        decisions,steps=[],[]
        for feature in features:
            common={k:feature[k] for k in ('branch_id','step','public_observation_sha256','policy_hidden_before_sha256','world_hidden_before_sha256','legal_mask','original_action')}
            decisions.append({**common,'probabilities':[{'probability':p} for p in feature['probabilities']],
                'final_action':feature['selected_action'],'candidate_actions':feature['guard_candidates']})
            steps.append({**common,'selected_action_hidden_sha256':feature['by_action_hidden_sha256'][str(feature['selected_action'])],
                'policy_hidden_after_sha256':feature['next_policy_hidden_sha256']})
        result=analysis.validate_runtime_evidence(features,decisions,steps,json.loads(runner.MANIFEST.read_text(encoding='utf-8')),status,
            json.loads((output/'runtime-costs.json').read_text(encoding='utf-8')),gate)
        assert result['model_call_counts']==dict(policy_encode=48,world_candidate_batch=48,actor_readout=50)


def test_execute_authorized_stub_retains_failed_staging_and_stops_at_order(monkeypatch):
    with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
        output = Path(temp_dir) / "failed-run"
        with pytest.raises(RuntimeError, match="stub branch failure at order 5"):
            _run_stubbed_execute(monkeypatch, output, fail_at=5)

        assert (output / "runtime-costs.json").is_file()
        assert (output / "run-status.json").is_file()
        assert (output / "failure-ledger.jsonl").is_file()
        assert not (output / "runner-staging" / "parent-00__W1__seed-1101__prefix-0__repeat-2__event_features_off").exists()
        failure_rows = [json.loads(line) for line in (output / "failure-ledger.jsonl").read_text(encoding="utf-8").splitlines()]
        assert any(row.get("reason") == "stub failure" for row in failure_rows)
        costs = json.loads((output / "runtime-costs.json").read_text(encoding="utf-8"))
        assert costs["status"] == "stopped_on_technical_error"
        status = json.loads((output / "run-status.json").read_text(encoding="utf-8"))
        assert status["status"] == "stopped_on_technical_error"
