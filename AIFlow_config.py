"""AIFlow 平台统一入口 —— 按任务名关键字自动分发（10-10 改造，参考 JENA 同款）。

═══════════════════════════════════════════════════════════════════════
为什么有这个文件（FastWAM 后缀式 + 平台只认固定名的矛盾）
═══════════════════════════════════════════════════════════════════════
平台只认固定名 AIFlow_config.py，且实测是 pod 启动时才从 userPath 拷贝并
import。「提交前 cp 改名」的 dance 在多任务并行排队时会串味：两个排队任务
读的是同一个文件，谁后启动谁吃错配置。

修复：本文件常驻不动，按任务名关键字**约定式自动分发**（新任务零改本文件）。
平台会设环境变量 JinnTrainResult = <userPath>/jinnTrainResult/<任务名>-<jobid>
（import 时已就位），取 basename 归一化（大写→小写、bench→eval 同义词）后，
扫描本目录的 AIFlow_config_<后缀>.py：后缀按下划线分词、**分词全部命中任务
名即为候选，最长后缀胜出**（长 = 更具体）：

    任务名含 eval + robotwin → AIFlow_config_eval_robotwin（RoboTwin ours 评测，任务名如：leapbot_ce_eval_robotwin）
    任务名含 robotwin        → AIFlow_config_robotwin    （RoboTwin 训练）
    任务名含 eval            → AIFlow_config_eval_libero （LIBERO ours 评测）
    任务名含 libero          → AIFlow_config_libero      （LIBERO 训练）
    无 JinnTrainResult（本机裸 import）→ 同 eval_libero（与改造前本文件
                                          内容一致，行为不变）
    有任务名但零匹配 → 直接 ImportError 列出可用后缀 —— 绝不静默吃错配置

从此提交流程 = 起一个含对应关键字的任务名即可，零改名，可任意并行排队：
    leapbot_ce_bench_libero / leapbot_ce_train_robotwin / leapbot_ce_eval_robotwin …

新加一类任务 = 新建 AIFlow_config_<新后缀>.py 放本目录（接口四件套：
__C = edict(); cfg = __C; __C.TRAIN.CMD = _build_cmd();
def AutoTrain(**params) -> bool，smoke 开关 AI_FLOW_* 写在该文件常改区），
任务名带上 <新后缀> 关键字即可，本文件不用动。

平台对接面不变：from AIFlow_config import cfg + AutoTrain(**params)，
本模块 import 时即解析出真实模式并原样转发（连 GPU_NUM 一并转发）。
"""

import importlib
import os
import sys

# 平台可能以 file-location 方式加载本文件（不一定把所在目录放进 sys.path），
# 兄弟模块分发 import 前显式补上 —— __file__ 指向平台拷贝的副本，自洽。
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_TASK_RAW = os.path.basename(os.environ.get("JinnTrainResult", ""))
# bench≡eval 同义归一：既有任务名 leapbot_ce_bench_libero 走评测分支
_TASK = _TASK_RAW.lower().replace("bench", "eval")

_DEFAULT_VARIANT = "eval_libero"


def _variants():
    for fn in os.listdir(_HERE):
        if fn.startswith("AIFlow_config_") and fn.endswith(".py"):
            yield fn[len("AIFlow_config_"):-len(".py")]


def _pick_variant():
    """返回 (后缀, 依据说明)。最长后缀胜出，并列取字母序并告警。"""
    if not _TASK_RAW:
        return _DEFAULT_VARIANT, "无 JinnTrainResult（本机裸 import）→ 缺省"
    cands = sorted(
        (-len(suffix), suffix)
        for suffix in _variants()
        if all(tok in _TASK for tok in suffix.split("_") if tok)
    )
    if not cands:
        avail = sorted(_variants())
        raise ImportError(
            f"[AIFlow_config] 任务名 '{_TASK_RAW}'（归一化 '{_TASK}'）不匹配任何"
            f"配置后缀，拒绝静默吃错配置。可用后缀：{avail}。"
            "任务名需含某个后缀的分词全部关键字（如 leapbot_ce_eval_robotwin）。"
        )
    if len(cands) > 1 and cands[0][0] == cands[1][0]:
        print(f"[AIFlow_config] 警告：后缀并列取字母序第一 "
              f"{cands[0][1]} vs {cands[1][1]}", flush=True)
    return cands[0][1], f"task '{_TASK_RAW}'"


_VARIANT, _WHY = _pick_variant()
print(f"[AIFlow_config] {_WHY} -> AIFlow_config_{_VARIANT}", flush=True)

_mod = importlib.import_module(f"AIFlow_config_{_VARIANT}")
AutoTrain = _mod.AutoTrain  # noqa: F401 平台入口
cfg = _mod.cfg  # noqa: F401 平台读 __C.TRAIN.CMD 走这里
__C = _mod.__C  # noqa: F401
GPU_NUM = getattr(_mod, "GPU_NUM", os.environ.get("GPU_NUM", "1"))  # noqa: F401
