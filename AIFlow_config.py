"""AIFlow 平台统一入口 —— 按任务名关键字自动分发（10-10 改造，参考 JENA 同款）。

═══════════════════════════════════════════════════════════════════════
为什么有这个文件（FastWAM 后缀式 + 平台只认固定名的矛盾）
═══════════════════════════════════════════════════════════════════════
平台只认固定名 AIFlow_config.py，且实测是 pod 启动时才拷贝并 import。
「提交前 cp 改名」的 dance 在多任务并行排队时会串味：两个排队任务读的是
同一个文件，谁后启动谁吃错配置。

修复：本文件常驻不动，按任务名关键字**约定式自动分发**（新任务零改本文件）。
平台会设环境变量 JinnTrainResult = <userPath>/jinnTrainResult/<任务名>-<jobid>
（import 时已就位），取 basename 归一化（大写→小写、bench→eval 同义词）后，
扫描 AIFlow_config_<后缀>.py：后缀按下划线分词、**分词全部是任务名（剥除
尾部 -<jobid> 后按 _/- 切词）的完整词即为候选，最长后缀胜出**（长 = 更具体）：

    任务名含 eval + robotwin → AIFlow_config_eval_robotwin（RoboTwin ours 评测，任务名如：leapbot_ce_eval_robotwin）
    任务名含 robotwin        → AIFlow_config_robotwin    （RoboTwin 训练）
    任务名含 eval            → AIFlow_config_eval_libero （LIBERO ours 评测）
    任务名含 libero          → AIFlow_config_libero      （LIBERO 训练）
    其余（含无 JinnTrainResult 的本机裸 import）→ 缺省 eval_libero

⚠️ 三条平台实锤（10-10 任务 21821436 / 10-13 任务 21932579 平台日志），分发器必须同时满足：
① **pod 端只把 AIFlow_config.py 单文件拷进任务目录**（/jinnMirror/.../
   2653853-<id>/，兄弟变体不在，os.listdir 扫出空）—— 变体查找必须依次兜底
   userPath env → cwd → 本树绝对根（/dahuafs 共享盘路径稳定，非机器依赖）；
② **import 本文件永不抛异常** —— 平台 caller 把 config import 包在
   try/except 里，任何异常 → conf 落回平台默认管线 → 找 dh_aiflow → 报
   「aiflowconfig.py有错误」秒挂。未匹配任务名只打横幅落缺省，横幅首行
   可见、可事后审计，绝不 raise。
③ **任务目录名尾部拼 -<jobid 8位十六进制>，哈希会撞关键字子串**（10-13
   实锤：a2 重提任务 ...a2_release-26fa02d4，"26fa02d4" 恰含子串 "a0"，
   子串匹配让 libero_a0_release 混进候选、与 libero_a2_release 同长并列、
   字母序取胜 → a2 任务静默跑成 a0 配置，还与真 a0 任务同写一个 run 目录）。
   修复 = 剥尾哈希 + 后缀分词必须是任务名的**完整词**（词集合等值匹配），
   子串命中一律不算数。

从此提交流程 = 起一个含对应关键字的任务名即可，零改名，可任意并行排队：
    leapbot_ce_bench_libero / leapbot_ce_train_robotwin / leapbot_ce_eval_robotwin …

新加一类任务 = 新建 AIFlow_config_<新后缀>.py 放本树根（接口四件套：
__C = edict(); cfg = __C; __C.TRAIN.CMD = _build_cmd();
def AutoTrain(**params) -> bool，smoke 开关 AI_FLOW_* 写在该文件常改区），
任务名带上 <新后缀> 关键字即可，本文件不用动。

平台对接面不变：from AIFlow_config import cfg + AutoTrain(**params)，
本模块 import 时即解析出真实模式并原样转发（连 GPU_NUM 一并转发）。
"""

import importlib
import os
import re
import sys

# 变体查找目录：__file__ 所在目录（pod 任务目录副本）→ userPath（平台注入，
# 变体所在的真实树根）→ cwd → 本树绝对根（兜底 env 未设；共享盘路径稳定）。
_HERE = os.path.dirname(os.path.abspath(__file__))
_TREE_ROOT = "/dahuafs/userdata/2653853/code/FastWAM"
_SEARCH_DIRS = [_HERE]
for _extra in (os.environ.get("userPath"), os.getcwd(), _TREE_ROOT):
    if _extra and all(os.path.realpath(_extra) != os.path.realpath(_d)
                      for _d in _SEARCH_DIRS):
        _SEARCH_DIRS.append(_extra)
for _d in _SEARCH_DIRS:
    if _d not in sys.path:
        sys.path.insert(0, _d)

_TASK_RAW = os.path.basename(os.environ.get("JinnTrainResult", ""))
# bench≡eval 同义归一：既有任务名 leapbot_ce_bench_libero 走评测分支
_TASK = _TASK_RAW.lower().replace("bench", "eval")
# 剥掉平台拼在末尾的 -<jobid 8位十六进制>（见文件头 ⚠️③：哈希子串会撞
# a0/a1/… 关键字，把任务劫持到错误变体）。剥不掉也无妨——下面的整词匹配
# 让哈希只算一个词，参与不了关键字命中。
_TASK_NAME = re.sub(r"-[0-9a-f]{8}$", "", _TASK)
# 整词集合：后缀的每个分词必须是任务名的完整词（等值），杜绝 "a0" 藏在
# "26fa02d4" 或其他词内部的子串误命中。
_TASK_WORDS = frozenset(w for w in re.split(r"[_\-]+", _TASK_NAME) if w)

_DEFAULT_VARIANT = "eval_libero"


def _variants():
    """收集搜索目录里全部 AIFlow_config_<后缀>.py 后缀（去重保序）。"""
    seen = set()
    for d in _SEARCH_DIRS:
        try:
            names = os.listdir(d)
        except OSError:
            continue
        for fn in names:
            if fn.startswith("AIFlow_config_") and fn.endswith(".py"):
                suffix = fn[len("AIFlow_config_"):-len(".py")]
                if suffix not in seen:
                    seen.add(suffix)
                    yield suffix


def _variant_file(suffix):
    """返回后缀对应的真实文件路径（按搜索顺序第一个命中），没有则 None。"""
    for d in _SEARCH_DIRS:
        p = os.path.join(d, f"AIFlow_config_{suffix}.py")
        if os.path.isfile(p):
            return d
    return None


def _pick_variant():
    """返回 (后缀, 目录, 依据说明)。最长后缀胜出，并列取字母序并告警；零匹配
    落缺省（绝不抛 —— 平台把 import 异常一律折成「aiflowconfig.py有错误」）。"""
    if not _TASK_RAW:
        d = _variant_file(_DEFAULT_VARIANT)
        return _DEFAULT_VARIANT, d, "无 JinnTrainResult（本机裸 import）→ 缺省"
    cands = sorted(
        (-len(suffix), suffix)
        for suffix in _variants()
        if all(tok in _TASK_WORDS for tok in suffix.split("_") if tok)
    )
    if not cands:
        d = _variant_file(_DEFAULT_VARIANT)
        return _DEFAULT_VARIANT, d, (
            f"任务名 '{_TASK_RAW}' 无匹配后缀 → 缺省 {_DEFAULT_VARIANT}"
            f"（可用后缀: {sorted(_variants())}；若非本意请核对任务名关键字）"
        )
    if len(cands) > 1 and cands[0][0] == cands[1][0]:
        print(f"[AIFlow_config] 警告：后缀并列取字母序第一 "
              f"{cands[0][1]} vs {cands[1][1]}", flush=True)
    suffix = cands[0][1]
    return suffix, _variant_file(suffix), f"task '{_TASK_RAW}'"


_VARIANT, _VARIANT_DIR, _WHY = _pick_variant()
print(f"[AIFlow_config] {_WHY} -> AIFlow_config_{_VARIANT}", flush=True)

if _VARIANT_DIR is None:
    # 理论不可达（树根兜底必有变体）；真发生时给全量诊断再让模块名错误自然抛。
    print(f"[AIFlow_config] 致命：变体文件在所有候选目录都找不到 "
          f"（搜索: {_SEARCH_DIRS}）", flush=True)
_mod = importlib.import_module(f"AIFlow_config_{_VARIANT}")
AutoTrain = _mod.AutoTrain  # noqa: F401 平台入口
cfg = _mod.cfg  # noqa: F401 平台读 __C.TRAIN.CMD 走这里
__C = _mod.__C  # noqa: F401
GPU_NUM = getattr(_mod, "GPU_NUM", os.environ.get("GPU_NUM", "1"))  # noqa: F401
