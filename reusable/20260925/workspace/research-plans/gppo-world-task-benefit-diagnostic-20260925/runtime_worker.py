"""Frozen runtime-necessity worker for the authorized stage-3 diagnostic.

The worker reuses the byte-for-byte stage-2 worker for model loading, public
environment handling, reward labels, and weight identity checks.  The three
inference kinds differ only at decision time:

* ``full`` uses the native online world candidate path;
* ``reference`` keeps the native actor evaluation path but neutralizes the
  candidate head with the independent hook helper;
* ``candidate`` removes the candidate head and never evaluates the world.

The latter two are frozen interventions on the P checkpoint.  They are not
policies trained without a world model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

from attribution_contract import authorize
from base_worker import Worker as BaseWorker
from cost_replay import groups, replay_batch
from decision_only_inference import actor_for_decision, world_for_decision
from no_online_world import actor_without_candidate, native_reference_without_candidate
from runtime_support import (
    PREP,
    ROOT,
    PublicMemory,
    canonical,
    ending,
    peak_rss,
    public_copy,
    read,
    timed,
    verify_vector,
    write,
)


KINDS = frozenset(("full", "reference", "candidate"))
REPLAY_STAGE = "A"
ENVIRONMENT_STAGES = frozenset(("B", "C"))
REPLAY_PATH = Path("runs/technical-gate-v1/decisions.jsonl")


class Worker(BaseWorker):
    """Stage-3 worker with explicit frozen online-world interventions."""

    def __init__(self, permit, binding, out):
        kind = permit.get("kind")
        stage = permit.get("stage")
        if kind != "candidate" or stage not in ("B", "C"):
            raise ValueError("Only N in B/C is authorized; A replay is forbidden")
        if kind not in KINDS:
            raise ValueError("runtime-necessity worker kind must be full, reference, or candidate")
        if kind in ("full", "reference") and stage != REPLAY_STAGE:
            raise ValueError("full/reference workers are restricted to replay stage A")
        if kind == "candidate" and stage not in {REPLAY_STAGE, "B", "C"}:
            raise ValueError("candidate worker has an unknown stage")

        super().__init__(permit, binding, out)
        self.world_guard_handle = None
        if kind in ("reference", "candidate"):
            self._install_inactive_world_guard()

    def _install_inactive_world_guard(self):
        """Fail closed if an intervention accidentally forwards the world."""

        register = getattr(self.world, "register_forward_pre_hook", None)
        if register is None:
            return

        def reject_world_forward(*_args, **_kwargs):
            raise RuntimeError("inactive world object was unexpectedly forwarded")

        self.world_guard_handle = register(reject_world_forward)

    def _tensor_sha256(self, value):
        """Hash a detached tensor after the timed decision has returned."""

        tensor = value.detach().cpu()
        if not bool(self.torch.isfinite(tensor).all().item()):
            raise RuntimeError("Nonfinite equivalence tensor")
        return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()

    def _tensor_values(self, value):
        tensor = value.detach().cpu()
        if not bool(self.torch.isfinite(tensor).all().item()):
            raise RuntimeError("Nonfinite output tensor")
        return tensor.tolist()

    def _select_action(self, probabilities, public, candidates):
        values = probabilities[0]
        if values.ndim != 1 or int(values.shape[0]) != 25:
            raise RuntimeError("Nonfinite or malformed probabilities")
        if not bool(self.torch.isfinite(values).all().item()):
            raise RuntimeError("Nonfinite or malformed probabilities")
        if bool((values < 0.0).any().item()):
            raise RuntimeError("Negative action probability")
        total = float(values.sum().detach().cpu())
        if not math.isclose(total, 1.0, rel_tol=1e-5, abs_tol=1e-5):
            raise RuntimeError("Action probabilities are not normalized")

        mask = list(public["mask"])
        legal = [int(action) for action in candidates if 0 <= int(action) < 25 and bool(mask[int(action)])]
        if not legal:
            raise RuntimeError("No legal replay action")
        action = max(legal, key=lambda item: (float(values[item].detach().cpu()), -item))
        if not bool(mask[action]):
            raise RuntimeError("Policy selected an illegal action")
        return action

    def choose(self, public, memory, state, candidates):
        """Run one frozen decision and return raw tensors for outer hashing."""

        ph, wh = (None, None) if state is None else state
        tensor = self.torch
        with tensor.no_grad():
            self.counts["policy_encode"] += 1
            features, pair_messages, next_policy_hidden = self.policy.encode(
                tensor.as_tensor(public["flat"], dtype=tensor.float32).reshape(1, -1), ph,
            )

            mask = tensor.as_tensor(public["mask"], dtype=tensor.bool)[None, :]
            if self.kind == "full":
                self.counts["world_candidate_batch"] += 1
                candidate_features, by_action = world_for_decision(
                    self.world, features, wh, public, use_events=True,
                )
                evaluation = actor_for_decision(
                    self.policy, features, pair_messages, self.pref, candidate_features, mask,
                )
                # The full branch needs the selected recurrent world state for
                # exact reproduction of the historical M decisions.
                action = self._select_action(evaluation["probabilities"][0:1], public, candidates)
                selected_world_hidden = by_action[action]["hidden"].detach()
                next_state = (next_policy_hidden.detach(), selected_world_hidden)
            elif self.kind == "reference":
                evaluation = native_reference_without_candidate(
                    self.policy, features, pair_messages, self.pref, mask,
                )
                action = self._select_action(evaluation["probabilities"][0:1], public, candidates)
                selected_world_hidden = None
                next_state = (next_policy_hidden.detach(), None)
            else:
                evaluation = actor_without_candidate(
                    self.policy, features, pair_messages, self.pref, mask,
                )
                action = self._select_action(evaluation["probabilities"][0:1], public, candidates)
                selected_world_hidden = None
                next_state = (next_policy_hidden.detach(), None)

            self.counts["actor_readout"] += 1
            bundle = {
                "logits": evaluation["logits"].detach(),
                "probabilities": evaluation["probabilities"].detach(),
                "policy_hidden": next_policy_hidden.detach(),
                "candidates": tuple(int(item) for item in candidates),
            }
            if selected_world_hidden is not None:
                bundle["selected_world_hidden"] = selected_world_hidden
            return action, next_state, bundle

    def _replay_inputs(self):
        path = PREP / REPLAY_PATH
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        traces = groups(rows, "M")
        expected = {(row["parent"], row["step"]): row for episode in traces for row in episode}
        if len(expected) != 25:
            raise RuntimeError("Frozen M replay must contain exactly 25 public decisions")
        return traces, expected

    def replay_once(self):
        """Replay the fixed 25-decision trace once; the parent owns repetition."""

        if self.stage != REPLAY_STAGE:
            raise RuntimeError("replay_once is restricted to stage A")
        traces, expected = self._replay_inputs()
        stats, records = replay_batch(traces, self.choose)
        summarized = []
        for parent, step, action, diagnostic, latency in records:
            row = expected[(parent, step)]
            candidate_actions = [int(item) for item in diagnostic["candidates"]]
            if self.kind == "full":
                if int(action) != int(row["action"]):
                    raise RuntimeError("Full replay action mismatch")
                if candidate_actions != [int(item) for item in row["candidates"]]:
                    raise RuntimeError("Full replay candidate-guard mismatch")
            if "selected_world_hidden" in diagnostic:
                world_hidden_sha = self._tensor_sha256(diagnostic["selected_world_hidden"])
                expected_world_sha = row["diagnostic"].get("selected_world_hidden_sha256")
                if world_hidden_sha != expected_world_sha:
                    raise RuntimeError("Full replay world-hidden mismatch")
            else:
                world_hidden_sha = None

            record = {
                "parent": parent,
                "step": int(step),
                "action": int(action),
                "candidates": candidate_actions,
                "wall_seconds": float(latency),
                "logits_sha256": self._tensor_sha256(diagnostic["logits"]),
                "probabilities_sha256": self._tensor_sha256(diagnostic["probabilities"]),
                "policy_hidden_sha256": self._tensor_sha256(diagnostic["policy_hidden"]),
            }
            if world_hidden_sha is not None:
                record["world_hidden_sha256"] = world_hidden_sha
            summarized.append(record)

        if len(summarized) != 25:
            raise RuntimeError("Replay returned an unexpected decision count")
        return {
            "stats": {
                "decisions": int(stats["decisions"]),
                "cpu_seconds": float(stats["cpu_seconds"]),
                "wall_seconds": float(stats["wall_seconds"]),
            },
            "records": summarized,
            "counts": dict(self.counts),
        }

    def _step_diagnostic(self, diagnostic):
        """Convert raw tensors to JSON only after the timed control block."""

        result = {
            "probabilities": self._tensor_values(diagnostic["probabilities"])[0],
            "logits": self._tensor_values(diagnostic["logits"])[0],
            "policy_hidden_sha256": self._tensor_sha256(diagnostic["policy_hidden"]),
            "world_enabled": self.kind == "full",
        }
        if "selected_world_hidden" in diagnostic:
            result["selected_world_hidden_sha256"] = self._tensor_sha256(
                diagnostic["selected_world_hidden"],
            )
        return result

    def reset(self, args):
        if self.kind != "candidate" or self.stage not in ENVIRONMENT_STAGES:
            raise RuntimeError("reset is restricted to candidate workers in stages B/C")
        return super().reset(args)

    def one_step(self):
        """Run the native B/C environment path without a fabricated world hash."""

        if self.kind != "candidate" or self.stage not in ENVIRONMENT_STAGES:
            raise RuntimeError("step is restricted to candidate workers in stages B/C")
        if self.done or self.step_index >= 18:
            raise RuntimeError("Invalid extra environment step")

        def control():
            public = public_copy(self.obs)
            self.memory.observe(public)
            candidates = self.memory.candidates(public)
            action, state, diagnostic = self.choose(public, self.memory, self.state, candidates)
            self.state = state
            self.memory.submitted(public, action)
            return public, candidates, action, diagnostic

        (public, candidates, action, diagnostic), controller = timed(control)
        step_diagnostic = self._step_diagnostic(diagnostic)

        self.counts["environment_steps"] += 1
        (self.obs, scalar, self.done, info), environment = timed(
            self.env.step, action, submit_command=True,
        )
        native_end = ending(self.done, info)
        (vector, counts, energy), labels = timed(
            verify_vector, info, self.prevcounts, self.energy, self.reward, self.config,
        )
        self.prevcounts, self.energy = counts, energy
        self.utility += 0.99 ** self.step_index * (0.4 * vector[0] + 0.2 * vector[1])
        if self.env.view._completed_tasks:
            raise RuntimeError("single_shot marker changed")
        row = {
            "step": self.step_index,
            "public": public,
            "candidates": candidates,
            "action": action,
            "submit_command": True,
            "diagnostic": step_diagnostic,
            "controller": controller,
            "environment": environment,
            "labels": labels,
            "vector_reward": vector,
            "scalar_reward": float(scalar),
            "done": bool(self.done),
            "info": info,
            "counts": dict(self.counts),
        }
        self.step_index += 1
        if self.step_index == 18 and not self.done:
            raise RuntimeError("Non-native truncation")
        if self.done:
            records = info["completion_records"]
            row["episode"] = {
                "steps": self.step_index,
                "utility": self.utility,
                "energy_used": 36 - self.energy,
                "task_counts": counts,
                "physical_on_time": sum(
                    value["physical_arrival_before_deadline"] is True
                    for value in records.values()
                ) / 6,
                "host_on_time_observed": sum(
                    value["host_confirmation_before_deadline"] is True
                    for value in records.values()
                ) / 6,
                "physical_completion_without_host_observed": sum(
                    value["host_confirmation_time"] is None
                    for value in records.values()
                ),
                **native_end,
            }
        return row


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--name", required=True)
    args = parser.parse_args(argv)

    worker = None
    try:
        _authorization, _binding, _request = authorize(args.authorization)
        out = ROOT / "runs/task-benefit-v1"
        permit = read(out / (args.name + "-permit.json"))
        write(out / (args.name + "-claim.json"), {"pid": os.getpid(), "permit": permit})
        worker = Worker(permit, read(PREP / "binding.json"), out)
        print(canonical({
            "ok": True,
            "ready": True,
            "counts": worker.counts,
            "sources": worker.sources,
            "initializer": "export-only shim",
            "torch_loaded": "torch" in sys.modules,
            "startup_cpu": time.process_time(),
            "kind": worker.kind,
            "stage": worker.stage,
        }), flush=True)
        for line in sys.stdin:
            if not line.strip():
                continue
            request = json.loads(line)
            command = request["command"]
            if command == "replay_once":
                raise RuntimeError("A replay is not authorized")
            elif command == "generate":
                value = worker.generate(request)
            elif command == "reset":
                value = worker.reset(request)
            elif command == "step":
                value = worker.one_step()
            elif command == "shutdown":
                if worker.weight_hash() != worker.before:
                    raise RuntimeError("Weights changed")
                value = {
                    "counts": dict(worker.counts),
                    "weights_unchanged": True,
                    "peak_rss": peak_rss(),
                    "kind": worker.kind,
                    "stage": worker.stage,
                }
            else:
                raise ValueError("Unknown worker command")
            print(canonical({"ok": True, "result": value}), flush=True)
            if command == "shutdown":
                return
    except BaseException as exc:
        print(canonical({
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "counts": None if worker is None else dict(worker.counts),
        }), flush=True)
        raise


if __name__ == "__main__":
    main()
