"""AIFlow 跨树委托变体：StarWAM D60/D80 时延探针（10-31 用户拍板）。

任务名含 eval + robotwin + starwam + ext → 本文件（4 词后缀，长于既有
eval_robotwin_starwam 3 词委托，最长后缀胜出），例如：
    leapbot_ce_eval_robotwin_starwam_ext

背景：RoboTwin v2 四窗基准里 StarWAM-MoT 保留率 80.4%（chunk 保护段内），
本探针把时延推出其 chunk 保护段（D60/D80），与 ours B6 探针单成对比较。
冻结口径：v1 既有 10 任务子集；对照窗 D{0,10,20,40} 取既有 v2 per-task
数据算同 10 任务受限 macro，不重跑。

与 eval_robotwin_starwam 委托的两处差异：
  ① 转发前预设 AI_FLOW_DELAYS / AI_FLOW_TASKS env —— StarWAM config 在
    import 期读这两个 env，缺省是全量 50 任务 × 四窗（v2 基准），必须
    在其 exec 前钉住探针口径（平台注入同名 env 时以 env 为准）。
  ② 转发后把模块常量 V2_SWEEP_ROOT 改指探针独立结果根
    starwam/robotwin_v2_sweep_ext 并重建 CMD —— 该常量不是 env 可覆盖项，
    缺省会写进 v2 主根（与既有 .done 幂等/聚合语义撞车）。AutoTrain 在
    调用时用 _build_cmd() 重建命令，故补丁后入口行为与展示 CMD 一致。

⚠️ 平台把 config import 异常一律折成「aiflowconfig.py有错误」秒挂 ——
   本文件 import 期绝不 raise：转发失败时打印横幅 + 给出秒败 AutoTrain，
   绝不静默回落 ours 配置。
"""

from __future__ import annotations

import importlib.util
import os
import sys

_BANNER = "[AIFlow_config_eval_robotwin_starwam_ext]"
_SIBLING = "/dahuafs/userdata/2653853/code/StarWAM/AIFlow_config.py"
_EXT_ROOT = ("/mnt/c2-worldmodel/2653853/baseline-evaluations/"
             "starwam/robotwin_v2_sweep_ext")

try:
    from easydict import EasyDict as edict
except ImportError:  # 与 sibling config 同款兜底
    class edict(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__

print(f"{_BANNER} starwam D60/D80 探针: 预设 env 后转发 {_SIBLING}", flush=True)

# ① 探针口径（sibling 在 import 期消费；平台注入同名 env 时以 env 为准）
os.environ.setdefault("AI_FLOW_DELAYS", "60 80")
os.environ.setdefault(
    "AI_FLOW_TASKS",
    # v1 既有 10 任务子集（与 ours B6 探针侧同表）
    "adjust_bottle beat_block_hammer blocks_ranking_rgb blocks_ranking_size "
    "click_alarmclock click_bell dump_bin_bigbin grab_roller handover_block "
    "handover_mic",
)

try:
    _spec = importlib.util.spec_from_file_location("_starwam_ext_real_config",
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
