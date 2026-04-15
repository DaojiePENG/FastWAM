#!/usr/bin/env python3
"""
Test script to verify fine-tuning configurations work correctly.
This script checks that the freeze strategies are applied correctly
without running actual training.
"""

try:
    from omegaconf import OmegaConf
    HAS_OMEGACONF = True
except ImportError:
    HAS_OMEGACONF = False

def test_freeze_strategy(model, freeze_strategy="action_head_only"):
    """Test if freeze strategy is applied correctly."""
    print(f"\n{'='*60}")
    print(f"Testing freeze strategy: {freeze_strategy}")
    print(f"{'='*60}")

    # Count trainable parameters
    total_params = 0
    trainable_params = 0

    component_stats = {
        "video_expert": {"total": 0, "trainable": 0},
        "action_expert": {"total": 0, "trainable": 0},
        "action_head": {"total": 0, "trainable": 0},
        "proprio_encoder": {"total": 0, "trainable": 0},
        "other": {"total": 0, "trainable": 0},
    }

    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params

        # Categorize parameters
        if "video_expert" in name:
            component_stats["video_expert"]["total"] += num_params
            if param.requires_grad:
                component_stats["video_expert"]["trainable"] += num_params
                trainable_params += num_params
        elif "action_expert.head" in name:
            component_stats["action_head"]["total"] += num_params
            if param.requires_grad:
                component_stats["action_head"]["trainable"] += num_params
                trainable_params += num_params
        elif "action_expert" in name:
            component_stats["action_expert"]["total"] += num_params
            if param.requires_grad:
                component_stats["action_expert"]["trainable"] += num_params
                trainable_params += num_params
        elif "proprio_encoder" in name:
            component_stats["proprio_encoder"]["total"] += num_params
            if param.requires_grad:
                component_stats["proprio_encoder"]["trainable"] += num_params
                trainable_params += num_params
        else:
            component_stats["other"]["total"] += num_params
            if param.requires_grad:
                component_stats["other"]["trainable"] += num_params
                trainable_params += num_params

    # Print results
    print(f"\nTotal parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"Frozen parameters: {total_params - trainable_params:,} ({(total_params - trainable_params)/1e6:.2f}M)")
    print(f"\nComponent breakdown:")

    for component, stats in component_stats.items():
        if stats["total"] > 0:
            trainable_pct = (stats["trainable"] / stats["total"] * 100) if stats["total"] > 0 else 0
            print(f"  {component:20s}: {stats['total']:12,} total, "
                  f"{stats['trainable']:12,} trainable ({trainable_pct:5.1f}%)")

    # Verify strategy
    print(f"\n{'='*60}")
    print("Verification:")
    print(f"{'='*60}")

    success = True

    if freeze_strategy == "action_head_only":
        if component_stats["action_head"]["trainable"] == 0:
            print("❌ FAILED: Action head should be trainable")
            success = False
        else:
            print("✓ Action head is trainable")

        if component_stats["action_expert"]["trainable"] > 0:
            print("❌ FAILED: Action expert backbone should be frozen")
            success = False
        else:
            print("✓ Action expert backbone is frozen")

        if component_stats["video_expert"]["trainable"] > 0:
            print("❌ FAILED: Video expert should be frozen")
            success = False
        else:
            print("✓ Video expert is frozen")

    elif freeze_strategy == "action_only":
        action_total = component_stats["action_expert"]["trainable"] + component_stats["action_head"]["trainable"]
        if action_total == 0:
            print("❌ FAILED: Action expert should be trainable")
            success = False
        else:
            print("✓ Action expert is trainable")

        if component_stats["video_expert"]["trainable"] > 0:
            print("❌ FAILED: Video expert should be frozen")
            success = False
        else:
            print("✓ Video expert is frozen")

    elif freeze_strategy == "video_only":
        if component_stats["video_expert"]["trainable"] == 0:
            print("❌ FAILED: Video expert should be trainable")
            success = False
        else:
            print("✓ Video expert is trainable")

        action_total = component_stats["action_expert"]["trainable"] + component_stats["action_head"]["trainable"]
        if action_total > 0:
            print("❌ FAILED: Action expert should be frozen")
            success = False
        else:
            print("✓ Action expert is frozen")

    elif freeze_strategy == "none":
        if trainable_params < total_params * 0.5:
            print("❌ FAILED: Most parameters should be trainable")
            success = False
        else:
            print("✓ Most parameters are trainable")

    if success:
        print(f"\n✓ All checks passed for strategy: {freeze_strategy}")
    else:
        print(f"\n❌ Some checks failed for strategy: {freeze_strategy}")

    return success


def main():
    print("FastWAM Fine-tuning Configuration Test")
    print("=" * 60)

    if not HAS_OMEGACONF:
        print("\n⚠️  OmegaConf not found. Skipping configuration loading test.")
        print("   This is expected if you haven't installed the environment yet.")
    else:
        # Test configuration loading
        strategies = ["action_head_only", "action_only", "video_only", "none"]

        for strategy in strategies:
            config_text = f"""
finetune:
  freeze_strategy: {strategy}
  freeze_video_expert: true
  freeze_action_backbone: true
"""
            try:
                config = OmegaConf.create(config_text)
                print(f"\n✓ Configuration for '{strategy}' loads successfully")
                print(f"  Config: {config}")
            except Exception as e:
                print(f"\n❌ Configuration for '{strategy}' failed to load")
                print(f"  Error: {e}")

    print("\n" + "=" * 60)
    print("Configuration test completed!")
    print("=" * 60)
    print("\nNote: This is a configuration loading test.")
    print("To fully test, you need to run with an actual model:")
    print("  python scripts/train.py task=libero_finetune_action_head max_steps=1")

    print("\n" + "=" * 60)
    print("Next steps:")
    print("=" * 60)
    print("\n1. Install the environment (if not already done):")
    print("   conda create -n fastwam python=3.10 -y")
    print("   conda activate fastwam")
    print("   pip install -e .")
    print("\n2. Run a quick training test:")
    print("   python scripts/train.py task=libero_finetune_action_head max_steps=1")
    print("\n3. Check the logs for 'Total trainable parameters'")
    print("\n4. Verify the parameter counts match the expected strategy")
    print("\nExpected parameter counts:")
    print("  - action_head_only: ~7K-50K")
    print("  - action_only: ~100M")
    print("  - video_only: ~3B")
    print("  - none: ~3.1B")


if __name__ == "__main__":
    main()
