# LeapBot-VA 脚本入口

正式 v7 实验只使用以下入口：

- `train_leapbot.sh`：一个 causal mode 的可配置 W strict-replay D30 或 multi-exit 训练；
- `train_causal_modes.sh`：相同合同依次训练三个 causal mode；
- `train_multi_exit.sh`：从获胜 D30 checkpoint 训练 D8/D16/D24/D30；
- `evaluate_checkpoint.sh`：单 checkpoint 的 LIBERO-Long 评测；
- `evaluate_pch_checkpoint.sh`：通过 `task=libero_leapbot_pch` 启动的 PCH 单 checkpoint 严格窗口评测；
- `evaluate_causal_modes.sh`：三模式和 FastWAM baseline 对比；
- `evaluate_pareto.sh`：固定为 checkpoint 训练 W 后比较出口深度并汇总 Pareto；
- `evaluate_fastwam_baseline.sh`：原始 FastWAM release baseline；
- `download_leapbot_assets.py`、`precompute_text_embeds.py`：正式资产与语言 cache；
- `validate_leapbot_checkpoint.py`、`build_eval_fingerprint.py`：训练/评测合同验收；
- `validate_real_6b_runtime_training_equivalence.py`：真实模型训练/在线因果路径 smoke。

核心配置：

```text
history_training_mode=strict_replay_window_bptt
history_window_blocks=8
episode_anchor=single_real_v0
history_padding=left_masked
learning_rate=1.0e-4
```

正式 launcher 使用 Accelerate + DeepSpeed ZeRO-2，并默认从脚本位置解析仓库根目录。HPC 预装环境可设置：

```bash
export PYTHON_BIN=/path/to/python
export ACCELERATE_BIN=/path/to/accelerate
```

详细顺序、Slurm 调试和 W&B 指标见 [TRAINING_AND_REPRODUCTION.md](../docs/TRAINING_AND_REPRODUCTION.md)。

PCH 前向优化、checkpoint 兼容规则和 task-config 评测合同见
[PCH_FORWARD_OPTIMIZATION_AND_TASK_CONFIG_EVALUATION.md](../docs/PCH_FORWARD_OPTIMIZATION_AND_TASK_CONFIG_EVALUATION.md)。
