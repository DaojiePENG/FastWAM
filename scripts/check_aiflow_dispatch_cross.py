#!/usr/bin/env python3
"""FastWAM 树跨树委托变体分发自测（10-19 四步铁律之①②；10-26 错单催生）。

背景：AHA/StarWAM 两单连续两次以 userPath=FastWAM 提交（22529775/22531435
与重提 22533775/22533786），平台 import 本树 ours eval config → 34 秒假完成。
修复 = 委托变体 AIFlow_config_eval_robotwin_{starwam,ahawam}.py（最长后缀
胜出，转发对应树真实 config）。

本 checker **完全模拟事故条件**：userPath=FastWAM（事故 userPath）+
JinnTrainResult 含真实哈希尾。断言：
  ① starwam/ahawam 任务路由到委托变体，CMD 带对应树驱动与 v2 旋钮；
  ② CMD 绝不含 ours 驱动签名（eval_leapbotce_robotwin_sweep = 错路由病原）；
  ③ AHA 工作目录 cd 到真实 RoboTwin 根（userPath=FastWAM 不得劫持）；
  ④ ours b6v2 回归：不被新 3 词后缀劫持。全绿 exit 0。

用法：
    <python> scripts/check_aiflow_dispatch_cross.py
"""

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (任务名, 期望变体后缀, CMD 必含子串, CMD 必不含子串)
CASES = [
    # StarWAM v2 全量（哈希尾用真单 22531435 的 3319081a）：委托 + StarWAM 驱动
    ("leapbot_ce_eval_robotwin_starwam_v2-3319081a", "eval_robotwin_starwam",
     ["STARWAM_SWEEP_ROOT=/mnt/c2-worldmodel/2653853/baseline-evaluations/"
      "starwam/robotwin_v2_sweep",
      "STARWAM_V2_DELAYS='0 10 20 40'",
      "STARWAM_V2_TASKS='beat_block_hammer",
      "starwam_robotwin_v2_chains_2653853.sh"],
     ["eval_leapbotce_robotwin_sweep", "LEAPBOTCE_"]),
    # StarWAM smoke：单格 place_shoe × D0（shlex 不给无空格值加引号）
    ("leapbot_ce_eval_robotwin_starwam_v2_smoke-ab12cd34",
     "eval_robotwin_starwam",
     ["STARWAM_V2_DELAYS=0", "STARWAM_V2_TASKS=place_shoe"],
     ["eval_leapbotce_robotwin_sweep"]),
    # AHA v2 全量（哈希尾用真单 22529775 的 4b907efa）：
    # cd 必须落真实 RoboTwin 根 —— userPath=FastWAM 不得劫持工作目录
    ("leapbot_ce_eval_robotwin_ahawam_v2-4b907efa", "eval_robotwin_ahawam",
     ["cd /dahuafs/userdata/2653853/code/RoboTwin",
      "run_sweep_cluster_xpolicy_2653853.sh", "AHA_WAM",
      "XPOLICY_SWEEP_SUFFIX=robotwin_v2_sweep",
      "ROBOTWIN_EVAL_DELAYS='0 10 20 40'",
      "ROBOTWIN_EVAL_TASKS='adjust_bottle beat_block_hammer"],
     ["eval_leapbotce_robotwin_sweep", "LEAPBOTCE_"]),
    # ours b6v2 回归：不得被新 3 词后缀劫持（整词匹配,b6v2 无 ahawam/starwam 词）
    ("leapbot_ce_eval_robotwin_b6v2-2a36b54e", "eval_robotwin",
     ["LEAPBOTCE_SWEEP_ROOT=/mnt/c2-worldmodel/2653853/baseline-evaluations/"
      "leapbotce/robotwin_v2_sweep_b6",
      "ROBOTWIN_EVAL_DELAYS='0 10 20 40'"],
     ["run_sweep_cluster_xpolicy", "starwam_robotwin_v2_chains"]),
]

# 完整复刻平台事故条件：userPath=FastWAM（错单 userPath）
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
    env["userPath"] = ROOT          # ★ 事故条件：userPath=FastWAM
    env["GPU_NUM"] = "8"
    for k in ("AI_FLOW_DELAYS", "AI_FLOW_TASKS", "AI_FLOW_POLICY",
              "AI_FLOW_SWEEP_SUFFIX", "AI_FLOW_RUN_DIR", "AI_FLOW_SWEEP_ROOT",
              "AI_FLOW_CKPT_LABEL"):
        env.pop(k, None)
    proc = subprocess.run([sys.executable, "-c", SNIP], env=env,
                          capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"import 失败 rc={proc.returncode}\n{proc.stderr[-800:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> int:
    print(f"[check] root={ROOT}\n[check] python={sys.executable}")
    failed = 0
    for name, want_variant, must, must_not in CASES:
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
