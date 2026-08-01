import inspect

from leapbot_va.models.leapbot import LeapBotVA


def test_memory_inference_uses_transient_future_video_without_vae_decode():
    source = inspect.getsource(LeapBotVA.infer_action)
    assert "infer_video_scheduler" in source
    assert "future_condition_kv" in source
    assert "_decode_latents" not in source
    assert "infer_joint" not in source
    assert "first_frame_latents" in source
    assert "append_observation" in source
    assert "append_actions" not in source


def test_prediction_is_transient_until_explicit_commit():
    inference_source = inspect.getsource(LeapBotVA.infer_action)
    commit_source = inspect.getsource(LeapBotVA.commit_executed_actions)
    assert "append_actions" not in inference_source
    assert "append_actions" in commit_source
