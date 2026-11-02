"""AIFlow 跨树委托变体：StarWAM chunk 剂量-响应探针（replan_steps=8）。

任务名含 eval + robotwin + starwam + ext + rs8 → 本文件（5 词后缀，压过
starwam_ext 4 词与 starwam 3 词，最长后缀胜出），例如：
    leapbot_ce_eval_robotwin_starwam_ext_rs8

背景（2026-10-31 用户拍板）：与 ours 侧 b6_ext_rs8 成对 —— 固定 D、chunk
执行步数 24→8，delay 采样频率 ×3（~16.6 → ~56 replans/集），画 retention
vs replan 频率的剂量曲线。判读矩阵见 b6_ext_rs8 docstring（双向可证伪）。
实现链：AI_FLOW_REPLAN_STEPS → StarWAM config._build_cmd 拼进
STARWAM_REPLAN_STEPS env → chains → lane 透传 → adapter
(XPolicyLab/policy/StarWAM/model.py) 覆盖 self.replan_steps。
模型与 delay 协议零改动，只改执行策略。

与 eval_robotwin_starwam_ext 委托的三处差异：
  ① 转发前预设 AI_FLOW_DELAYS / AI_FLOW_TASKS / AI_FLOW_REPLAN_STEPS env
    —— StarWAM config 在 import 期消费这三个 env（前两者缺省 = 全量 50
    任务 × 四窗；REPLAN_STEPS 缺省 = 空 = 官方 24 步），必须在 exec 前钉住
    探针口径（平台注入同名 env 时以 env 为准）。
  ② 转发后把模块常量 V2_SWEEP_ROOT 改指探针独立结果根
    starwam/robotwin_v2_sweep_ext_rs8 并重建 CMD（同 ext 委托 ②）。
  ③ CMD 断言含 STARWAM_REPLAN_STEPS=8（checker 用例锚定）。

⚠️ 平台把 config import 异常一律折成「aiflowconfig.py有错误」秒挂 ——
   本文件 import 期绝不 raise：转发失败时打印横幅 + 给出秒败 AutoTrain，
   绝不静默回落 ours 配置。
"""

from __future__ import annotations

import importlib.util
import os
import sys

_BANNER = "[AIFlow_config_eval_robotwin_starwam_ext_rs8]"
_SIBLING = "/dahuafs/userdata/2653853/code/StarWAM/AIFlow_config.py"
_EXT_ROOT = ("/mnt/c2-worldmodel/2653853/baseline-evaluations/"
             "starwam/robotwin_v2_sweep_ext_rs8")

try:
    from easydict import EasyDict as edict
except ImportError:  # 与 sibling config 同款兜底
    class edict(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__

print(f"{_BANNER} starwam chunk=8 剂量探针: 预设 env 后转发 {_SIBLING}", flush=True)

# ① 探针口径（sibling 在 import 期消费；平台注入同名 env 时以 env 为准）
# 11-02 用户指示：chunk8 补全六窗 D{0,10,20,40,60,80}；既有 D{0,40} 两窗
# 已收齐（.done 幂等跳过不重跑），实际新增 D{10,20,60,80} 40 格。
os.environ.setdefault("AI_FLOW_DELAYS", "0 10 20 40 60 80")
os.environ.setdefault("AI_FLOW_REPLAN_STEPS", "8")  # 剂量变体本体
os.environ.setdefault(
    "AI_FLOW_TASKS",
    # v1 既有 10 任务子集（与 ours 探针侧同表）
    "adjust_bottle beat_block_hammer blocks_ranking_rgb blocks_ranking_size "
    "click_alarmclock click_bell dump_bin_bigbin grab_roller handover_block "
    "handover_mic",
)

try:
    _spec = importlib.util.spec_from_file_location("_starwam_ext_rs8_real_config",
                                                   _SIBLING)
    if _spec is None or _spec.loader is None:
        raise ImportError(f"spec_from_file_location 失败: {_SIBLING}")
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _mod
    _spec.loader.exec_module(_mod)

    # ② 独立结果根：patch 模块常量后重建 CMD（模块级 CMD 已按旧根生成，
    #    AutoTrain 调用时重建的命令与此处保持一致）
    _mod.V2_SWEEP_ROOT = _EXT_ROOT
    _cmd = _mod._build_cmd()
    _mod.cfg.TRAIN.CMD = _cmd
    _mod.cfg.CMD = _cmd

    cfg = _mod.cfg                # noqa: F401 平台 from AIFlow_config import cfg
    AutoTrain = _mod.AutoTrain    # noqa: F401 平台入口
    __C = getattr(_mod, "__C", None) or cfg  # noqa: F401 dispatcher 取 __C
    GPU_NUM = getattr(_mod, "GPU_NUM",
                      os.environ.get("GPU_NUM", "1"))  # noqa: F401
    print(f"{_BANNER} 转发 OK, CMD={str(cfg.TRAIN.CMD)[:160]}", flush=True)
except Exception as _exc:  # noqa: BLE001 —— 绝不 raise（见文件头 ⚠️）
    print(f"{_BANNER} 错误: 转发失败: {_exc!r}", flush=True)
    __C = edict()
    cfg = __C
    __C.FRAME = "general"
    __C.TRAIN = edict()
    __C.TRAIN.ENABLE = True
    __C.TRAIN.NET_TYPE = "generation"
    __C.TRAIN.CMD = f"echo {_BANNER} sibling-config-load-failed && exit 1"
    __C.CMD = __C.TRAIN.CMD

    def AutoTrain(**params) -> bool:  # noqa: F811 - 秒败兜底,原因可见
        print(f"{_BANNER} AutoTrain 拒绝执行: 委托目标加载失败: {_exc!r}",
              flush=True)
        return False

    GPU_NUM = os.environ.get("GPU_NUM", "1")  # noqa: F401
