"""AIFlow 跨树委托变体：任务名含整词 `starwam` → 转发 code/StarWAM 真实 config。

═══════════════════════════════════════════════════════════════════════
为什么有这个文件（2026-10-26 两次错单实录）
═══════════════════════════════════════════════════════════════════════
平台按 **userPath** 找 AIFlow_config.py，任务名前缀不参与路由。StarWAM 单
22531435 / 22533786 两次都以 userPath=FastWAM 提交 → 平台 import 本树的
AIFlow_config_eval_robotwin（ours）→ 跑成 ours 默认旧 sweep（旧 ckpt/5 窗/
旧根全 .done 秒跳）→ 34 秒假完成，StarWAM 代码零执行。

修复：按树根 dispatcher 的既有扩展模式（「新加一类任务 = 新建
AIFlow_config_<新后缀>.py 放本树根」），建**更长后缀**的委托变体 ——
任务名 leapbot_ce_eval_robotwin_starwam_v2 的分词含 eval+robotwin+starwam，
本后缀（3 词）按「最长后缀胜出」压过 eval_robotwin（2 词），import 期把
code/StarWAM/AIFlow_config.py 按真实路径加载并原样转发其 cfg / AutoTrain /
__C / GPU_NUM。StarWAM config 全部路径为共享盘绝对路径（CHAINS_SH、
V2_SWEEP_ROOT、双 env prefix），委托加载行为与 userPath=code/StarWAM 原生
提交完全一致；userPath 选 FastWAM / RoboTwin / StarWAM 都能正确路由。

⚠️ 平台把 config import 异常一律折成「aiflowconfig.py有错误」秒挂 ——
   本文件 import 期绝不 raise：转发失败时打印横幅 + 给出秒败 AutoTrain
   （任务快速失败、train.log 首屏可见原因），绝不静默回落 ours 配置
   （那正是本文件要消灭的错路由）。

命名冲突防御：sibling 以私有模块名加载（sys.modules 注册），不污染
AIFlow_config_* 命名空间；本变体不含 starwam 关键字以外的逻辑。
"""

from __future__ import annotations

import importlib.util
import os
import sys

_BANNER = "[AIFlow_config_eval_robotwin_starwam]"
_SIBLING = "/dahuafs/userdata/2653853/code/StarWAM/AIFlow_config.py"

try:
    from easydict import EasyDict as edict
except ImportError:  # 与 sibling config 同款兜底
    class edict(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__

print(f"{_BANNER} 任务名含整词 starwam -> 转发 {_SIBLING}", flush=True)

try:
    _spec = importlib.util.spec_from_file_location("_starwam_v2_real_config",
                                                   _SIBLING)
    if _spec is None or _spec.loader is None:
        raise ImportError(f"spec_from_file_location 失败: {_SIBLING}")
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _mod
    _spec.loader.exec_module(_mod)

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
