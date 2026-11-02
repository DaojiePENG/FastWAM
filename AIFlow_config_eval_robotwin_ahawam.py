"""AIFlow 跨树委托变体：任务名含整词 `ahawam` → 转发 code/RoboTwin 真实 config。

═══════════════════════════════════════════════════════════════════════
为什么有这个文件（2026-10-26 两次错单实录，与 starwam 委托变体同款）
═══════════════════════════════════════════════════════════════════════
AHA-WAM 单 22529775 / 22533775 两次都以 userPath=FastWAM 提交 → 平台
import 本树的 AIFlow_config_eval_robotwin（ours）→ 跑成 ours 默认旧 sweep
→ 34 秒假完成，AHA 代码零执行。

修复：任务名 leapbot_ce_eval_robotwin_ahawam_v2 的分词含
eval+robotwin+ahawam，本后缀（3 词）按 dispatcher「最长后缀胜出」压过
eval_robotwin（2 词），import 期把 code/RoboTwin/AIFlow_config.py 按真实
路径加载并原样转发其 cfg / AutoTrain / __C / GPU_NUM。

路径解析关键（已验证）：RoboTwin config 的 `_repo_workdir()` 先试 userPath、
本树根（FastWAM）**没有** run_sweep_cluster_xpolicy_2653853.sh（实锤 10-26）
→ 落回 `Path(__file__).parent` = 真实 RoboTwin 树根 → `cd <真实根> && …
bash run_sweep_cluster_xpolicy_2653853.sh AHA_WAM` 正确；v2 旋钮
（四窗 + 50 任务 + robotwin_v2_sweep 独立根）由其任务名含 "v2" 的内建
快捷键提供，userPath=FastWAM / RoboTwin 都能正确路由。

⚠️ 平台把 config import 异常一律折成「aiflowconfig.py有错误」秒挂 ——
   本文件 import 期绝不 raise：转发失败时打印横幅 + 秒败 AutoTrain，
   绝不静默回落 ours 配置（那正是本文件要消灭的错路由）。

═══════════════════════════════════════════════════════════════════════
D60/80 补窗内建（2026-11-02 用户拍板 + 平台限制实锤）
═══════════════════════════════════════════════════════════════════════
AHA rs16 v2 主表四窗已收齐 200 格（宏均 D0 93.5 / D10 78.7 / D20 54.3 /
D40 29.6），D40 > 10% 判据 → 六窗全量补 D{60,80}（100 格，同 v2 根）。
原方案「同名重提 + 平台注入 AI_FLOW_DELAYS env」不可行：平台提单系统
只能选卡数/卡型 + 任务命名，无 env 栏（11-02 实锤）→ 补窗口径直接烤进
本变体：转发前 setdefault AI_FLOW_DELAYS='60 80'。链路三段已验：
sibling import 期消费该 env（v2 分支仅空缺省时才落四窗）→ _build_cmd()
显式拼 ROBOTWIN_EVAL_DELAYS='60 80' 进 CMD → driver read 消费。
选纯 '60 80' 而非六窗缺省：xpolicy driver 无 skip-existing（只数 .done
做进度汇报），落六窗会把已收四窗 200 格全部重跑。
注意：本名此后重提永远只补 D{60,80}；若格已存在会重跑覆盖（driver 无
幂等），勿在收齐后重复提交同名单。
"""

from __future__ import annotations

import importlib.util
import os
import sys

_BANNER = "[AIFlow_config_eval_robotwin_ahawam]"
_SIBLING = "/dahuafs/userdata/2653853/code/RoboTwin/AIFlow_config.py"

try:
    from easydict import EasyDict as edict
except ImportError:  # 与 sibling config 同款兜底
    class edict(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__

print(f"{_BANNER} 任务名含整词 ahawam -> 转发 {_SIBLING}", flush=True)

# ① 补窗口径（sibling 在 import 期消费；平台无法注 env，此处为唯一入口）
# 11-02 拍板：D40=29.6% > 10% 判据 → 同 v2 根补 D{60,80} 全量 100 格；
# 四窗 200 格已收齐在既有结果里，绝不重跑（见文件头冻结段）。
os.environ.setdefault("AI_FLOW_DELAYS", "60 80")

try:
    _spec = importlib.util.spec_from_file_location("_ahawam_v2_real_config",
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
