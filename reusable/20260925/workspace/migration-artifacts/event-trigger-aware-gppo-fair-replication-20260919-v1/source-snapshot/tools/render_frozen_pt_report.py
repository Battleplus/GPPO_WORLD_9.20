"""Render the frozen P/T validation summary as a concise report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def pct(x):
    return "n/a" if x is None else f"{100*x:.2f}%"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--audit", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    s = json.loads(args.summary.read_text(encoding="utf-8"))
    a = json.loads(args.audit.read_text(encoding="utf-8"))
    lines = [
        "# 修复后 P/T 矩阵账本核对与冻结 validation",
        "",
        "本报告只读取已提交 checkpoint、commit、ledger、SQLite 预算账本和冻结 validation tape；本轮没有训练、optimizer 更新、补步或新 tape。",
        "",
        "## 恢复链与资格",
        "",
        "| run | 逻辑环境步 | 策略更新 | pending window | checkpoint SHA-256 | finite |",
        "|---|---:|---:|---:|---|---|",
    ]
    for name, info in a["models"].items():
        c = info["counters"]
        lines.append(f"| {name} | {c.get('environment_steps')} | {c.get('policy_optimizer_steps')} | {c.get('pending_window_k')} | `{info['checkpoint_sha256']}` | {info['finite']} |")
    lines += [
        "",
        "SQLite 全局账本：environment_steps reserved=49152, verified=49151, unknown=1, pending=0；policy_optimizer_calls reserved=379, verified=379；world_optimizer_calls=0。unknown 是 `formal-v4-T_train-3307` 的 1 个 environment-step reservation，保留为 unknown，未确认执行，也未退款。",
        "",
        "T/1101 的 128 步 checkpoint（SHA `8cc5dc446dd9588400eb9495e715c00a13b5b9da1be2b519b61595c14477f0b8`）被作为修复后恢复输入；第一次更新在历史续行重放处失败，未提交 optimizer 更新。修复后账本显示从该状态继续到逻辑 8191 步，增加的逻辑步为 8063；现有 commit/ledger 没有证据表明初始 128 步被再次晋升为正式训练样本。失败尾部未被拼接或退款。",
        "",
        "六组逻辑训练量不等：P1101=7707/58，T1101=8191/62，P2203=8192/63，T2203=8192/64，P3307=8192/63，T3307=8122/62（步/策略更新）。因此比较不是完全等预算的严格单变量实验。",
        "",
        "## 冻结 validation 完成度",
        "",
        f"432/432 episodes，6080 环境步；每个模型/seed/通信条件 16 episodes、96 tasks。独立 test 未读取；结果唯一键 432/432，无重复。",
        "",
        "| model | seed | I-ideal | W1-light | W2-moderate | W1/W2 等权 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in s["models"]:
        for seed in s["seeds"]:
            rates=[]
            vals=[]
            for c in s["conditions"]:
                v=s["table"][f"{model}|{seed}|{c}"]
                rates.append(v["completion_rate"]); vals.append(f"{v['completed']}/{v['tasks']}")
            lines.append(f"| {model} | {seed} | {vals[0]} ({pct(rates[0])}) | {vals[1]} ({pct(rates[1])}) | {vals[2]} ({pct(rates[2])}) | {pct((rates[1]+rates[2])/2)} |")
    lines += [
        "",
        "### 主/次比较",
        "",
    ]
    for key, label in (("main_T_triggered_minus_P_triggered_W1_W2_equal_weight", "主比较 T_train+T_dispatch − P_train+T_dispatch"), ("secondary_T_triggered_minus_P_periodic_W1_W2_equal_weight", "次比较 T_train+T_dispatch − P_train+周期")):
        v=s["comparisons"][key]
        lines.append(f"- {label}: 父 tape 平均差 {pct(v['mean'])}；10000 次父 tape bootstrap 95% CI [{pct(v['ci95'][0])}, {pct(v['ci95'][1])}]；独立父 tape={v['parents']}。区间只覆盖 tape 不确定性，不覆盖训练 seed 不确定性。")
    lines += [
        "- 主比较逐 seed 的 W1/W2 等权差：1101=+1.56 个百分点，2203=-0.52 个百分点，3307=+5.21 个百分点；2/3 seed 为正。",
        "- 按预登记开发门槛，点估计与父 tape 区间下界为正，但该矩阵训练量不等，故只能作为开发信号，不能作因果收益结论。",
        "",
        "## 任务、安全、成本",
        "",
        "全部 432 episodes 合计 2592 tasks：P_train+周期 653/864，P_train+T 648/864，T_train+T 660/864。未决任务 0；记录内非法新动作、非法续行和安全违规均为 0。",
        "",
        "| model | total energy | actor calls | continuation steps | world calls | decision mean/p95/p99 (episode-summary ms) |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for model in s["models"]:
        vals=[s["table"][f"{model}|{seed}|{c}"] for seed in s["seeds"] for c in s["conditions"]]
        e=sum(v["energy_used_sum"] for v in vals); actor=sum(v["actor_calls"] for v in vals); cont=sum(v["continuation_steps"] for v in vals); world=sum(v["world_calls"] for v in vals)
        t={key: sum(v["decision_timing_episode_summary"][key] for v in vals)/len(vals) for key in ("mean_ms", "p95_ms", "p99_ms")}
        lines.append(f"| {model} | {e:.3f} | {actor} | {cont} | {world} | {t['mean_ms']:.3f}/{t['p95_ms']:.3f}/{t['p99_ms']:.3f}; see JSON for all strata |")
    lines += [
        "",
        "通信代理字节在环境记录中不可用（432/432 rows 标记 bytes_observed=false），因此没有将其记为 0。decision timing 仅保存每 episode 的 mean/p95/p99 汇总，未保存原始逐决策样本；报告的 timing 为 episode-summary 聚合，不是精确 pooled quantile。",
        "",
        "## 文件",
        "",
        f"- 原始逐 episode ledger: `{args.summary.parent / 'episodes.jsonl'}`",
        f"- 评价协议: `{args.summary.parent / 'evaluation-protocol.json'}`",
        f"- 统计 JSON: `{args.summary}`",
        f"- 恢复链 audit: `{args.audit}`",
    ]
    args.out.write_text("\n".join(lines)+"\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
