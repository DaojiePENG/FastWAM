#!/usr/bin/env python3
"""FastWAM 树 AIFlow 分发自测（10-19 四步铁律之①；JENA scripts/check_aiflow_dispatch.py 同款）。

模拟平台 import：对每个任务名设 JinnTrainResult / userPath / GPU_NUM 后
真实 import 树根 AIFlow_config（dispatcher），断言落点变体与 CMD 关键串。
含 b6 快捷键的双向子串包含检查（铁律之②在 CASES 里体现：b6_d40 先于裸
d40 命中、train a6_lora 名不得被 b6 误伤）。全绿 exit 0。

用法：
    <python> scripts/check_aiflow_dispatch.py          # 用 sys.executable
    CHECK_PY=<env python> python scripts/check_aiflow_dispatch.py
"""

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (任务名, 期望 dispatcher 变体后缀, CMD 必含子串, CMD 必不含子串)
CASES = [
    # B 线默认提交：不得带 RUN_DIR/SWEEP_ROOT/EVAL_TASK（保持 B 线默认路由）
    ("leapbot_ce_eval_robotwin", "eval_robotwin",
     ["eval_leapbotce_robotwin_sweep_2653853.sh", "LEAPBOTCE_CKPT=latest"],
     ["LEAPBOTCE_RUN_DIR", "LEAPBOTCE_SWEEP_ROOT", "LEAPBOTCE_EVAL_TASK"]),
    # B6 全量：四旋钮齐 + 四窗默认（EVAL_TASK=codec_lora 配方，10-26 事故修复）
    ("leapbot_ce_eval_robotwin_b6", "eval_robotwin",
     ["LEAPBOTCE_RUN_DIR=/mnt/c2-worldmodel/2653853/leapbot-ce/runs/"
      "robotwin_all_leapbotce_codec_lora/robotwin_all_codec_lora_dtrain20_seed42_gpu_n3",
      "LEAPBOTCE_SWEEP_ROOT=/mnt/c2-worldmodel/2653853/baseline-evaluations/"
      "leapbotce/robotwin_sweep_b6",
      "ROBOTWIN_EVAL_DELAYS='0 10 20 40'",
      "LEAPBOTCE_EVAL_TASK=robotwin_all_leapbotce_codec_lora"],
     []),
    # b6 + 时延令牌：令牌优先（d40 ⊃ "d4" 无关，但 b6 仍须命中 run 切换）
    ("leapbot_ce_eval_robotwin_b6_d40", "eval_robotwin",
     ["ROBOTWIN_EVAL_DELAYS=40",
      "LEAPBOTCE_RUN_DIR=/mnt/c2-worldmodel/2653853/leapbot-ce/runs/"
      "robotwin_all_leapbotce_codec_lora",
      "LEAPBOTCE_EVAL_TASK=robotwin_all_leapbotce_codec_lora"],
     []),
    # 裸 d40（无 b6）：B 线默认，绝不得带 B6 run
    ("leapbot_ce_eval_robotwin_d40", "eval_robotwin",
     ["ROBOTWIN_EVAL_DELAYS=40"],
     ["LEAPBOTCE_RUN_DIR"]),
    # 训练名（含 a6_lora —— 与 b6 无子串包含，不得互相污染）
    ("leapbot_ce_train_robotwin_a6_lora", "train_robotwin_a6_lora", [], []),
    # LIBERO 薄委托变体：最长后缀胜出
    ("leapbot_ce_eval_libero_a1_ft_fresh100_full",
     "eval_libero_a1_ft_fresh100_full", ["libero"], []),
]

SNIP = r"""
import json, os, sys
sys.path.insert(0, os.environ["CHECK_USERPATH"])
import AIFlow_config as d
print(json.dumps({"variant": str(getattr(d, "_VARIANT", "?")),
                  "cmd": str(d.cfg.TRAIN.CMD)}))
"""


def run_case(name: str) -> dict:
    env = dict(os.environ)
    env["JinnTrainResult"] = f"/tmp/fake_jtr/{name}"
    env["CHECK_USERPATH"] = ROOT
    env["GPU_NUM"] = "8"
    env.pop("AI_FLOW_DELAYS", None)
    env.pop("AI_FLOW_TASKS", None)
    env.pop("AI_FLOW_RUN_DIR", None)
    env.pop("AI_FLOW_SWEEP_ROOT", None)
    proc = subprocess.run([sys.executable, "-c", SNIP], env=env,
                          capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"import 失败 rc={proc.returncode}\n{proc.stderr[-800:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> int:
    py = os.environ.get("CHECK_PY", sys.executable)
    print(f"[check] root={ROOT}\n[check] python={py}")
    failed = 0
    for name, want_variant, must, must_not in CASES:
        env = dict(os.environ)
        env["CHECK_USERPATH"] = ROOT
        try:
            info = run_case(name)
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {name}: {exc}")
            failed += 1
            continue
        problems = []
        if want_variant and info["variant"] != want_variant:
            problems.append(f"变体 {info['variant']} != 期望 {want_variant}")
        for s in must:
            if s not in info["cmd"]:
                problems.append(f"CMD 缺 {s!r}")
        for s in must_not:
            if s in info["cmd"]:
                problems.append(f"CMD 不应含 {s!r}")
        if problems:
            failed += 1
            print(f"FAIL {name}: " + "; ".join(problems))
        else:
            print(f"PASS {name} -> {info['variant']}")
        print(f"       CMD: {info['cmd'][:220]}{'…' if len(info['cmd']) > 220 else ''}")
    print(f"[check] {len(CASES) - failed}/{len(CASES)} PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
