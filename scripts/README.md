# LeapBot-VA 脚本入口

完整的环境、资产、训练、评测、资源和集群复现说明见
[`docs/TRAINING_AND_REPRODUCTION.md`](../docs/TRAINING_AND_REPRODUCTION.md)。

## 日常直接运行的入口

按正式流水线顺序：

1. `screen_learning_rate.sh`：两个学习率的全历史配对筛选。
2. `audit_learning_rate.sh`：固定样本/噪声审计并生成 LR 选择 manifest。
3. `screen_h0_retention.sh`：比较真实 episode-start 样本的 x1/x4 采样。
4. `audit_h0_retention.sh`：生成 H0 选择 manifest。
5. `train_causal_modes.sh`：从相同 FastWAM 权重训练三种 D30 causal mode。
6. `evaluate_causal_modes.sh`：统一运行 dev10 或 final50 模式对比。
7. `train_multi_exit.sh`：从获胜 D30 checkpoint 训练 D8/16/24/30 出口。
8. `evaluate_pareto.sh`：评测深度与 KV 保留上限网格。

按需入口：

- `train_leapbot.sh`：单个模式的 canonical ZeRO-2 训练器；其他正式训练入口最终调用它。
- `evaluate_checkpoint.sh`：评测一个 LeapBot checkpoint。
- `evaluate_fastwam_baseline.sh`：评测 FastWAM release baseline。
- `precompute_text_embeds.py`：正式训练前生成 T5 prompt cache。

这些入口都应从仓库根目录执行，并由实际拥有数据与 checkpoint 的用户运行。
不要把 `train.py` 当作正式入口：它是 Hydra/Trainer 内核，不包含完整的资产、GPU、
run-contract 和 checkpoint 验收保护。

## 正式入口会调用的支持工具

下列脚本不是旧调试残留，canonical 流水线会直接调用或在硬件验收时使用：

- `download_leapbot_assets.py`、`training_asset_manifest.py`、
  `verify_text_cache_provenance.py`：固定并验证数据、权重、VAE 与 T5 cache 身份。
- `validate_leapbot_checkpoint.py`、`validate_run_contract_group.py`、
  `build_eval_fingerprint.py`：阻止跨配置混用 checkpoint 或评测结果。
- `history_audit_selection.py`、`history_stratified_loss.py`：固定噪声历史审计与选择报告。
- `full_prefix_smoke.py`、`validate_real_6b_runtime_training_equivalence.py`：
  验证完整历史训练和在线 KV 路径的数值等价性。
- `preprocess_action_dit_backbone.py`：上游 FastWAM checkpoint 准备工具。

`accelerate_configs/` 与 `ds_configs/` 是 `train_leapbot.sh` 使用的正式 DeepSpeed 配置，
不是独立命令。`fastwam_legacy/` 仅为保留上游兼容性，不用于 LeapBot 正式实验；其中
launcher 对未验收的多节点配置会直接报错。

生成的 `__pycache__/` 不属于 Git 交付内容，可安全删除。
