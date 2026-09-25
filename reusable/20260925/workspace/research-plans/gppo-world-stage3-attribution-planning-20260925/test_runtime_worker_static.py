"""Static contract tests for the stage-3 runtime worker.

These tests deliberately parse source only.  They do not import the worker,
load a checkpoint, construct an environment, or import Torch/NumPy/native
model modules.
"""

from __future__ import annotations

import ast
from pathlib import Path
import unittest


SOURCE_PATH = Path(__file__).with_name("runtime_worker.py")


def _tree():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    return ast.parse(source), source


def _worker_class(tree):
    return next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Worker"
    )


def _method(worker, name):
    return next(
        node for node in worker.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )


def _dict_string_keys(node):
    keys = set()
    for candidate in ast.walk(node):
        if not isinstance(candidate, ast.Dict):
            continue
        for key in candidate.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                keys.add(key.value)
    return keys


def _called_names(node):
    names = set()
    for candidate in ast.walk(node):
        if not isinstance(candidate, ast.Call):
            continue
        function = candidate.func
        if isinstance(function, ast.Name):
            names.add(function.id)
        elif isinstance(function, ast.Attribute):
            names.add(function.attr)
    return names


class RuntimeWorkerStaticTests(unittest.TestCase):
    def test_source_compiles_without_execution(self):
        source = SOURCE_PATH.read_text(encoding="utf-8")
        compile(source, str(SOURCE_PATH), "exec")

    def test_no_forbidden_model_imports(self):
        tree, _ = _tree()
        forbidden = {"torch", "numpy", "gppo_world"}
        found = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.extend(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                found.append(node.module.split(".", 1)[0])
        self.assertTrue(forbidden.isdisjoint(found), found)

    def test_worker_subclasses_base_and_exposes_bounded_commands(self):
        tree, _ = _tree()
        worker = _worker_class(tree)
        self.assertIn("BaseWorker", {base.id for base in worker.bases if isinstance(base, ast.Name)})
        methods = {
            node.name for node in worker.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertTrue({"__init__", "choose", "replay_once", "reset", "one_step"}.issubset(methods))

    def test_replay_is_single_pass_and_has_parent_schema(self):
        tree, _ = _tree()
        replay = _method(_worker_class(tree), "replay_once")
        self.assertIn("replay_batch", _called_names(replay))
        self.assertIn("groups", _called_names(replay) | _called_names(_method(_worker_class(tree), "_replay_inputs")))
        for node in ast.walk(replay):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range":
                self.fail("replay_once must not loop over fixed repetitions")
        keys = _dict_string_keys(replay)
        self.assertTrue({"stats", "records", "counts"}.issubset(keys))
        self.assertTrue({
            "parent", "step", "action", "candidates", "wall_seconds", "logits_sha256",
            "probabilities_sha256", "policy_hidden_sha256",
        }.issubset(keys))

    def test_full_world_path_and_no_world_paths_are_explicit(self):
        tree, source = _tree()
        choose = _method(_worker_class(tree), "choose")
        calls = _called_names(choose)
        self.assertIn("world_for_decision", calls)
        self.assertIn("actor_for_decision", calls)
        self.assertIn("native_reference_without_candidate", calls)
        self.assertIn("actor_without_candidate", calls)
        self.assertIn('self.kind == "full"', source)
        self.assertIn('self.counts["world_candidate_batch"] += 1', source)
        self.assertIn('self.state = state', source)

    def test_candidate_environment_path_rejects_world_hash_fabrication(self):
        tree, source = _tree()
        reset = _method(_worker_class(tree), "reset")
        step = _method(_worker_class(tree), "one_step")
        reset_names = _called_names(reset)
        step_names = _called_names(step)
        self.assertIn("super", reset_names)
        self.assertIn("timed", step_names)
        self.assertIn("_step_diagnostic", step_names)
        self.assertIn('"world_enabled": self.kind == "full"', source)
        self.assertIn('"selected_world_hidden" in diagnostic', source)
        self.assertIn("self.kind != \"candidate\"", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
