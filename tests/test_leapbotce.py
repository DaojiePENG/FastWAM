import numpy as np
import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace

from fastwam.datasets.lerobot import cloudedge_robot_video_dataset as cloudedge_dataset
from fastwam.datasets.lerobot.cloudedge_robot_video_dataset import (
    CloudEdgeRobotVideoDataset,
    sample_stale_index,
)
from fastwam.models.wan22.leapbotce import (
    CloudPlanningCache,
    EdgeVisionEncoder,
    LeapBotCE,
    stale_loss_weight,
)
from fastwam.models.wan22.cloud_kv_adapter import CloudKVAdapter, CloudPlanningCodec
from fastwam.models.wan22 import leapbotce_edge_train_factory
from fastwam.trainer import Wan22Trainer


class TinyVision(nn.Module):
    embed_dim = 4

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, self.embed_dim)

    def forward(self, image):
        return self.proj(image.mean(dim=(-1, -2)))


def test_stale_loss_curriculum():
    assert stale_loss_weight(0, 100, 0.5) == 0.0
    assert stale_loss_weight(50, 100, 0.5) == pytest.approx(0.25)
    assert stale_loss_weight(200, 100, 0.5) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        stale_loss_weight(1, 10, 1.1)


def test_cloud_planning_codec_compresses_payload_and_supports_decoder():
    torch.manual_seed(0)
    cache_k = [torch.randn(2, 12, 6) for _ in range(3)]
    cache_v = [torch.randn(2, 12, 6) for _ in range(3)]

    codec = CloudPlanningCodec(num_layers=3, cache_dim=6, pool_grid=(2, 3), bottleneck_dim=8)
    out_k, out_v = codec(cache_k, cache_v, (2, 6))
    assert len(out_k) == 3 and len(out_v) == 3
    for key, value in zip(out_k, out_v):
        assert key.shape == (2, 6, 6)  # M = pool_grid product replaces the 12-token cache
        torch.testing.assert_close(key, value)  # no decoder: K == V at every layer
    assert out_k[0].dtype == cache_v[0].dtype

    codec_dec = CloudPlanningCodec(
        num_layers=3, cache_dim=6, pool_grid=(2, 3), bottleneck_dim=8,
        use_kv_decoder=True, decoder_rank=4)
    dec_k, dec_v = codec_dec(cache_k, cache_v, (2, 6))
    assert not torch.equal(dec_k[0], dec_v[0])  # decoder decouples addressing from content

    with pytest.raises(ValueError, match="layer count"):
        codec(cache_k[:1], cache_v[:1], (2, 6))
    with pytest.raises(ValueError, match="Token grid"):
        codec(cache_k, cache_v, (3, 6))
    with pytest.raises(ValueError, match="channel mismatch"):
        codec([torch.zeros(2, 12, 5) for _ in range(3)],
              [torch.zeros(2, 12, 5) for _ in range(3)], (2, 6))


def test_cloud_planning_codec_layer_mixture_is_softmax_and_learnable():
    codec = CloudPlanningCodec(num_layers=3, cache_dim=4, pool_grid=(1, 2), bottleneck_dim=4)
    with torch.no_grad():
        codec.layer_logits.copy_(torch.tensor([4.0, 0.0, 0.0]))
    cache_v = [torch.randn(1, 2, 4) * (idx + 1) for idx in range(3)]
    cache_k = [torch.zeros(1, 2, 4) for _ in range(3)]
    out_k, out_v = codec(cache_k, cache_v, (1, 2))
    weights = torch.softmax(torch.tensor([4.0, 0.0, 0.0]), dim=0)
    mixed = sum(w * v for w, v in zip(weights, cache_v))
    mlp = codec.out_proj(torch.nn.functional.gelu(codec.in_proj(codec.norm(mixed))))
    torch.testing.assert_close(out_v[0], mlp, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(out_k[0], mlp, atol=1e-6, rtol=1e-5)  # K == V without decoder
    out_v[0].sum().backward()
    assert codec.layer_logits.grad is not None
    assert codec.layer_logits.grad.abs().sum() > 0


def test_edge_encoder_select_mode_picks_cls_and_topk_patches():
    torch.manual_seed(0)

    class TinyTokenVision(nn.Module):
        embed_dim = 4
        num_prefix_tokens = 1

        def __init__(self):
            super().__init__()
            self.tok = nn.Linear(3, 4)

        def forward_features(self, image):
            batch = image.shape[0]
            cls = self.tok(image.mean(dim=(-1, -2))).unsqueeze(1)
            patches = self.tok(image.permute(0, 2, 3, 1).reshape(batch, -1, 3))
            return torch.cat([cls, patches], dim=1)

    backbone = TinyTokenVision()
    encoder = EdgeVisionEncoder(
        text_dim=8, num_views=2, encoder=backbone, embed_dim=4, freeze=True,
        mode="select", edge_dim=6, tokens_per_view=3, num_layers=2)
    views = torch.rand(2, 2, 3, 224, 224)
    output = encoder(views)
    assert output.shape == (2, 6, 6)  # 2 views x (1 cls + 2 patches), edge_dim 6

    flat = views.flatten(0, 1)
    flat = (flat - 0.5) / 0.5
    feats = backbone.forward_features(flat)
    cls = feats[:, 0]
    patches = feats[:, 1:]
    scores = torch.cosine_similarity(patches, cls.unsqueeze(1).expand_as(patches), dim=-1)
    topk = scores.topk(2, dim=1).indices
    expected_tokens = torch.cat(
        [feats[:, :1], torch.gather(patches, 1, topk.unsqueeze(-1).expand(-1, -1, 4))], dim=1)
    expected = encoder.projector.proj(expected_tokens.reshape(2, 6, 4))
    torch.testing.assert_close(output, expected)

    output.sum().backward()
    assert all(parameter.grad is None for parameter in backbone.parameters())
    assert encoder.projector.proj.weight.grad is not None

    # Per-layer K/V scales only enter through the prefix expansion path.
    model = LeapBotCE.__new__(LeapBotCE)
    nn.Module.__init__(model)
    model.mot = SimpleNamespace(num_layers=2)
    model.edge_vision = encoder
    prefix = encoder(views)
    prefix_k, prefix_v = model._edge_prefix_kv(prefix)
    (prefix_k[1].sum() + prefix_v[0].sum()).backward()
    assert encoder.projector.scale_k.grad is not None
    assert encoder.projector.scale_v.grad is not None


def test_edge_prefix_kv_expands_per_layer_scales():
    model = LeapBotCE.__new__(LeapBotCE)
    nn.Module.__init__(model)
    model.mot = SimpleNamespace(num_layers=2)
    model.edge_vision = SimpleNamespace(
        mode="select",
        projector=SimpleNamespace(scale_k=torch.tensor([1.0, 2.0]),
                                  scale_v=torch.tensor([0.5, 0.5])))
    prefix = torch.ones(1, 3, 4)
    prefix_k, prefix_v = model._edge_prefix_kv(prefix)
    assert len(prefix_k) == 2 and len(prefix_v) == 2
    torch.testing.assert_close(prefix_k[0], prefix)
    torch.testing.assert_close(prefix_k[1], 2.0 * prefix)
    torch.testing.assert_close(prefix_v[0], 0.5 * prefix)


def test_edge_encoder_grid_selection_is_uniform():
    torch.manual_seed(0)

    class TinyTokenVision(nn.Module):
        embed_dim = 4
        num_prefix_tokens = 1

        def __init__(self):
            super().__init__()
            self.tok = nn.Linear(3, 4)

        def forward_features(self, image):
            batch = image.shape[0]
            cls = self.tok(image.mean(dim=(-1, -2))).unsqueeze(1)
            patches = self.tok(image.permute(0, 2, 3, 1).reshape(batch, -1, 3))
            return torch.cat([cls, patches], dim=1)

    backbone = TinyTokenVision()
    encoder = EdgeVisionEncoder(
        text_dim=8, num_views=1, encoder=backbone, embed_dim=4, freeze=True,
        mode="select", edge_dim=6, tokens_per_view=3, num_layers=2,
        selection="grid")
    views = torch.rand(2, 1, 3, 224, 224)
    output = encoder(views)
    assert output.shape == (2, 3, 6)

    flat = views.flatten(0, 1)
    flat = (flat - 0.5) / 0.5
    feats = backbone.forward_features(flat)
    patches = feats[:, 1:]
    keep_idx = torch.linspace(0, patches.shape[1] - 1, 2).round().long()
    expected_tokens = torch.cat([feats[:, :1], patches[:, keep_idx]], dim=1)
    expected = encoder.projector.proj(expected_tokens.reshape(2, 3, 4))
    torch.testing.assert_close(output, expected)


def test_edge_encoder_select_mode_requires_explicit_edge_dim():
    # Regression: select-mode prefix K/V are concatenated with the video
    # cache K/V (heads*head_dim = 3072 on Wan2.2-TI2V-5B), so silently
    # defaulting edge_dim to text_dim (= 4096 there) crashed every
    # select-mode training at the first torch.cat.  Construction must fail
    # loudly instead.
    class TinyTokenVision(nn.Module):
        embed_dim = 4
        num_prefix_tokens = 1

    with pytest.raises(ValueError, match="edge_dim"):
        EdgeVisionEncoder(
            text_dim=8, num_views=2, encoder=TinyTokenVision(), embed_dim=4,
            freeze=True, mode="select", tokens_per_view=3, num_layers=2)


def test_edge_prefix_switch_gates_prefix_encoding():
    model = LeapBotCE.__new__(LeapBotCE)
    nn.Module.__init__(model)
    views = torch.zeros(1, 2, 3, 8, 8)
    model.edge_vision = SimpleNamespace(mode="select")
    model.edge_prefix_enabled = False
    assert model._encode_edge_prefix(views) is None
    model.edge_vision = SimpleNamespace(mode="pooled")
    assert model._encode_edge_prefix(views) is None


def test_codec_checkpoint_roundtrip_and_mode_mismatch(tmp_path):
    def make_shell(with_codec, mode="select"):
        model = LeapBotCE.__new__(LeapBotCE)
        nn.Module.__init__(model)
        model.action_expert = nn.Linear(3, 3)
        model.edge_vision = SimpleNamespace(projector=nn.Linear(3, 3), mode=mode)
        model.proprio_encoder = None
        model.cloud_kv_adapter = None
        model.cloud_kv_codec = (
            CloudPlanningCodec(num_layers=2, cache_dim=4, pool_grid=(1, 2))
            if with_codec else None)
        model.edge_num_views = 2
        model.stale_loss_lambda_max = 0.5
        model.stale_loss_warmup_steps = 1
        model.torch_dtype = torch.float32
        return model

    source = make_shell(with_codec=True)
    with torch.no_grad():
        source.cloud_kv_codec.scale_k.fill_(1.5)
        source.cloud_kv_codec.out_proj.bias.fill_(0.25)
    path = tmp_path / "codec_delta.pt"
    source.save_checkpoint(path)

    restored = make_shell(with_codec=True)
    restored.load_checkpoint(path)
    torch.testing.assert_close(restored.cloud_kv_codec.scale_k, source.cloud_kv_codec.scale_k)
    torch.testing.assert_close(restored.cloud_kv_codec.out_proj.bias, source.cloud_kv_codec.out_proj.bias)

    with pytest.raises(ValueError, match="cloud_kv_codec"):
        make_shell(with_codec=False).load_checkpoint(path)
    with pytest.raises(ValueError, match="edge_vision_mode"):
        make_shell(with_codec=True, mode="pooled").load_checkpoint(path)


def test_cloud_kv_adapter_is_identity_then_learns_residual():
    adapter = CloudKVAdapter(num_layers=2, cache_dim=6, rank=3)
    # The source cloud cache is treated as a frozen tensor.  Gradients should
    # still reach the lightweight interface adapter.
    keys = [torch.randn(2, 4, 6) for _ in range(2)]
    values = [torch.randn(2, 4, 6) for _ in range(2)]

    adapted_keys, adapted_values = adapter(keys, values)
    assert isinstance(adapted_keys, list)
    assert isinstance(adapted_values, list)
    for original, adapted in zip(keys, adapted_keys):
        torch.testing.assert_close(adapted, original)
        assert adapted.dtype == original.dtype
        assert adapted.shape == original.shape
    for original, adapted in zip(values, adapted_values):
        torch.testing.assert_close(adapted, original)

    # Zero-init up projections preserve the baseline, while still providing a
    # useful gradient to the adapter when the cache is frozen.
    loss = sum(item.square().mean() for item in adapted_keys + adapted_values)
    loss.backward()
    assert adapter.up_k.weight.grad is not None
    assert adapter.up_v.weight.grad is not None
    assert adapter.up_k.weight.grad.abs().sum() > 0
    assert adapter.up_v.weight.grad.abs().sum() > 0
    assert all(item.grad is None for item in keys + values)


def test_cloud_kv_adapter_validates_cache_contract():
    adapter = CloudKVAdapter(num_layers=2, cache_dim=4, rank=2)
    cache = [torch.zeros(1, 3, 4) for _ in range(2)]
    with pytest.raises(ValueError, match="layer count"):
        adapter(cache[:1], cache[:1])
    with pytest.raises(ValueError, match="shape mismatch"):
        adapter(cache, [torch.zeros(1, 2, 4) for _ in range(2)])
    with pytest.raises(ValueError, match="channel mismatch"):
        bad = [torch.zeros(1, 3, 5) for _ in range(2)]
        adapter(bad, bad)
    with pytest.raises(ValueError, match=r"shape \[batch, sequence, channels\]"):
        bad = [torch.zeros(1, 4) for _ in range(2)]
        adapter(bad, bad)


def test_cloud_kv_adapter_checkpoint_roundtrip(tmp_path):
    def make_shell():
        model = LeapBotCE.__new__(LeapBotCE)
        nn.Module.__init__(model)
        model.action_expert = nn.Linear(3, 3)
        model.edge_vision = SimpleNamespace(projector=nn.Linear(3, 3), mode="pooled")
        model.proprio_encoder = None
        model.cloud_kv_adapter = CloudKVAdapter(num_layers=2, cache_dim=4, rank=2)
        model.cloud_kv_codec = None
        model.edge_num_views = 2
        model.stale_loss_lambda_max = 0.5
        model.stale_loss_warmup_steps = 1
        model.torch_dtype = torch.float32
        return model

    source = make_shell()
    with torch.no_grad():
        source.cloud_kv_adapter.up_k.weight.fill_(0.125)
        source.cloud_kv_adapter.layer_scale_v.fill_(0.75)
    path = tmp_path / "leapbotce_delta.pt"
    source.save_checkpoint(path)

    restored = make_shell()
    restored.load_checkpoint(path)
    torch.testing.assert_close(
        restored.cloud_kv_adapter.up_k.weight,
        source.cloud_kv_adapter.up_k.weight,
    )
    torch.testing.assert_close(
        restored.cloud_kv_adapter.layer_scale_v,
        source.cloud_kv_adapter.layer_scale_v,
    )


def test_cloud_kv_adapter_is_in_dit_train_registry_without_unfreezing_cloud():
    class TinyTrainShell(nn.Module):
        def __init__(self):
            super().__init__()
            self.video_expert = nn.Linear(4, 4)
            self.cloud_kv_adapter = CloudKVAdapter(num_layers=2, cache_dim=4, rank=2)
            self.dit = nn.ModuleDict({"cloud_kv_adapter": self.cloud_kv_adapter})

    model = TinyTrainShell()
    Wan22Trainer._apply_dit_only_train_mode(model)
    assert all(not parameter.requires_grad for parameter in model.video_expert.parameters())
    assert all(parameter.requires_grad for parameter in model.cloud_kv_adapter.parameters())
    assert model.cloud_kv_adapter.training


def test_edge_factory_registers_optional_cloud_adapter(monkeypatch):
    class TinyLeapBotCE(nn.Module):
        def __init__(self):
            super().__init__()
            self.action_expert = nn.Linear(2, 2)
            self.edge_vision = SimpleNamespace(projector=nn.Linear(2, 2))
            self.cloud_kv_adapter = CloudKVAdapter(num_layers=1, cache_dim=4, rank=2)

    model = TinyLeapBotCE()
    monkeypatch.setattr(
        leapbotce_edge_train_factory,
        "_create_local_leapbotce",
        lambda **_kwargs: model,
    )
    output = leapbotce_edge_train_factory.create_leapbotce()
    assert output is model
    assert set(output.dit.keys()) == {
        "action_expert",
        "edge_vision_projector",
        "cloud_kv_adapter",
    }
    assert output.dit["cloud_kv_adapter"] is output.cloud_kv_adapter


def test_edge_factory_registers_planning_codec(monkeypatch):
    class TinyLeapBotCE(nn.Module):
        def __init__(self):
            super().__init__()
            self.action_expert = nn.Linear(2, 2)
            self.edge_vision = SimpleNamespace(projector=nn.Linear(2, 2))
            self.cloud_kv_adapter = None
            self.cloud_kv_codec = CloudPlanningCodec(
                num_layers=1, cache_dim=4, pool_grid=(1, 2))

    model = TinyLeapBotCE()
    monkeypatch.setattr(
        leapbotce_edge_train_factory,
        "_create_local_leapbotce",
        lambda **_kwargs: model,
    )
    output = leapbotce_edge_train_factory.create_leapbotce()
    assert output is model
    assert set(output.dit.keys()) == {
        "action_expert",
        "edge_vision_projector",
        "cloud_kv_codec",
    }
    assert output.dit["cloud_kv_codec"] is output.cloud_kv_codec


def test_cloud_encoding_applies_kv_adapter_at_cache_boundary():
    class TinyVideoExpert:
        fuse_vae_embedding_in_latents = False

        @staticmethod
        def prepare(latents, timestep, context, context_mask, *_args):
            del timestep
            batch = latents.shape[0]
            tokens = torch.zeros(batch, 3, 4)
            t_mod = torch.zeros(batch, 1, 4)
            freqs = torch.zeros(3, 1, 2, dtype=torch.complex64)
            return (tokens, None, t_mod, context, context_mask, freqs, None, None, None, 1)

        @staticmethod
        def build_video_to_video_mask(seq_len, tokens_per_frame, device):
            del tokens_per_frame
            return torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)

    class TinyMoT:
        num_heads = 1
        attn_head_dim = 4
        num_layers = 2
        raw_keys = None
        raw_values = None

        @classmethod
        def prefill_video_cache_tensor(cls, *_args):
            key = torch.randn(1, 3, 4)
            value = torch.randn(1, 3, 4)
            cls.raw_keys = [key.clone(), key.clone()]
            cls.raw_values = [value.clone(), value.clone()]
            return cls.raw_keys, cls.raw_values

    model = LeapBotCE.__new__(LeapBotCE)
    nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.torch_dtype = torch.float32
    model.video_expert = TinyVideoExpert()
    model.mot = TinyMoT()
    model._encode_video_latents = lambda image: torch.zeros(image.shape[0], 1, 1, 1, 1)
    model.cloud_kv_adapter = CloudKVAdapter(num_layers=2, cache_dim=4, rank=2)
    with torch.no_grad():
        model.cloud_kv_adapter.up_k.weight.fill_(0.1)
        model.cloud_kv_adapter.up_v.weight.fill_(0.1)

    context = torch.zeros(1, 1, 4)
    context_mask = torch.ones(1, 1, dtype=torch.bool)
    cache = model._encode_cloud_tensor(
        torch.zeros(1, 3, 8, 8), context, context_mask
    )
    assert len(cache.video_cache_k) == 2
    assert cache.video_cache_k[0].shape == (1, 3, 4)
    assert cache.video_cache_k[0].dtype == torch.float32
    assert not torch.equal(cache.video_cache_k[0], TinyMoT.raw_keys[0])
    assert not torch.equal(cache.video_cache_v[0], TinyMoT.raw_values[0])


def test_stale_index_respects_padding_and_boundary():
    pad = torch.tensor([True, True, False, False, False])
    stale, delay = sample_stale_index(4, 4, pad, np.random.RandomState(0))
    assert stale in {2, 3}
    assert delay == 4 - stale
    assert sample_stale_index(0, 20, torch.tensor([False]), np.random.RandomState(0)) == (0, 0)


def test_dataset_keeps_cloud_image_and_proprio_time_aligned(monkeypatch):
    dataset = CloudEdgeRobotVideoDataset.__new__(CloudEdgeRobotVideoDataset)
    dataset.max_delay_steps = 2
    dataset.action_horizon = 2
    dataset.concat_multi_camera = "horizontal"
    sample = {
        "video": torch.arange(5, dtype=torch.float32).view(1, 5, 1, 1).expand(3, 5, 1, 2),
        "image_is_pad": torch.zeros(5, dtype=torch.bool),
        "proprio": torch.arange(10, dtype=torch.float32).view(5, 2),
        "action": torch.zeros(5, 7),
        "action_is_pad": torch.zeros(5, dtype=torch.bool),
        "prompt": "task",
    }
    dataset.dataset = [sample]
    monkeypatch.setattr(cloudedge_dataset, "sample_stale_index", lambda *_: (0, 2))

    output = dataset[0]

    torch.testing.assert_close(output["cloud_stale_image"], sample["video"][:, 0])
    torch.testing.assert_close(output["cloud_stale_proprio"], sample["proprio"][0:1])
    torch.testing.assert_close(output["cloud_current_image"], sample["video"][:, 2])
    torch.testing.assert_close(output["cloud_current_proprio"], sample["proprio"][2:3])
    torch.testing.assert_close(output["edge_current_proprio"], sample["proprio"][2:3])


def test_edge_encoder_freezes_backbone_and_trains_projector():
    backbone = TinyVision()
    encoder = EdgeVisionEncoder(
        text_dim=8, num_views=2, encoder=backbone, embed_dim=4, freeze=True
    )
    output = encoder(torch.rand(2, 2, 3, 32, 32))
    assert output.shape == (2, 1, 8)
    output.sum().backward()
    assert all(parameter.grad is None for parameter in backbone.parameters())
    assert encoder.projector[0].weight.grad is not None
    encoder.train()
    assert not backbone.training


class TinyInferenceScheduler:
    def build_inference_schedule(self, num_steps, device, dtype, shift_override=None):
        del shift_override
        return torch.ones(num_steps, device=device, dtype=dtype), torch.full(
            (num_steps,), -0.25, device=device, dtype=dtype
        )

    @staticmethod
    def step(prediction, delta, action):
        return action + prediction * delta


def test_cloud_cache_uses_cloud_state_and_retains_language_only_context():
    model = LeapBotCE.__new__(LeapBotCE)
    nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.torch_dtype = torch.float32
    model.proprio_dim = 2
    model.proprio_encoder = nn.Linear(2, 4, bias=False)
    with torch.no_grad():
        model.proprio_encoder.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]]))

    language = torch.ones(1, 2, 4)
    language_mask = torch.ones(1, 2, dtype=torch.bool)
    captured = {}

    def encode_prompt(_):
        return language, language_mask

    def encode_tensor(image, cloud_context, cloud_mask, edge_context=None, edge_mask=None):
        captured["image"] = image
        captured["cloud_context"] = cloud_context
        captured["cloud_mask"] = cloud_mask
        return CloudPlanningCache([], [], edge_context, edge_mask, 1, 1)

    model.encode_prompt = encode_prompt
    model._encode_cloud_tensor = encode_tensor
    cloud_proprio = torch.tensor([[2.0, 3.0]])
    cache = model.encode_cloud(torch.zeros(1, 3, 8, 8), prompt="task", cloud_proprio=cloud_proprio)

    assert captured["cloud_context"].shape == (1, 3, 4)
    torch.testing.assert_close(
        captured["cloud_context"][:, -1], model.proprio_encoder(cloud_proprio)
    )
    assert captured["cloud_mask"].shape == (1, 3)
    torch.testing.assert_close(cache.context, language)
    torch.testing.assert_close(cache.context_mask, language_mask)


def test_sync_wrapper_matches_two_stage_edge_inference_with_fixed_seed():
    # Build a tiny inference-only shell around the real two-stage methods. This
    # validates the wrapper contract without loading the 5B frozen cloud expert.
    model = LeapBotCE.__new__(LeapBotCE)
    nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.torch_dtype = torch.float32
    model.edge_num_views = 2
    model.edge_vision = SimpleNamespace(mode="pooled", projector=nn.Linear(8, 4))
    model.action_expert = SimpleNamespace(action_dim=2)
    model.infer_action_scheduler = TinyInferenceScheduler()
    model.proprio_dim = 8
    model.proprio_encoder = nn.Linear(8, 4, bias=False)

    context = torch.ones(1, 2, 4)
    context_mask = torch.ones(1, 2, dtype=torch.bool)
    cache = CloudPlanningCache([], [], context, context_mask, video_seq_len=1, video_tokens_per_frame=1)
    observed = {}

    def encode_cloud(input_image, prompt=None, context=None, context_mask=None,
                     cloud_proprio=None, proprio=None):
        observed["input_shape"] = tuple(input_image.shape)
        observed["prompt"] = prompt
        observed["cloud_proprio"] = cloud_proprio if cloud_proprio is not None else proprio
        return cache

    def edge_context(cache_context, cache_mask, current_views):
        assert current_views.shape == (1, 2, 3, 8, 4)
        assert cache_context.shape == (1, 3, 4)
        observed["edge_context"] = cache_context
        return cache_context, cache_mask

    def predict_from_cache(cache_arg, noisy_action, timestep, edge_context_arg, edge_mask,
                           edge_prefix=None):
        assert cache_arg is cache
        assert timestep.shape == (1,)
        assert edge_context_arg is observed["edge_context"]
        assert edge_mask.shape == (1, 3)
        assert edge_prefix is None  # pooled edge vision carries no prefix K/V
        return torch.full_like(noisy_action, 0.5)

    model.encode_cloud = encode_cloud
    model._edge_context = edge_context
    model._predict_from_cache = predict_from_cache
    image = torch.rand(1, 3, 8, 8)
    proprio = torch.rand(1, 8)

    two_stage = model.infer_action_edge(
        cache, model._split_views(image, 2), action_horizon=3, edge_proprio=proprio,
        num_inference_steps=2, seed=7
    )
    synchronous = model.infer_action(
        "task", image, action_horizon=3, proprio=proprio, num_inference_steps=2, seed=7
    )

    assert observed["input_shape"] == (1, 3, 8, 8)
    assert observed["prompt"] == "task"
    assert observed["cloud_proprio"] is proprio
    assert observed["edge_context"].shape == (1, 3, 4)
    assert two_stage["action"].dtype == torch.float32
    torch.testing.assert_close(synchronous["action"], two_stage["action"])


def test_cloud_cache_switch_skips_cloud_forward():
    # "w/o cloud cache" (eval-time): encode_cloud must not touch the cloud
    # expert at all and must return a zero-length planning cache that still
    # carries the language context for the edge side.
    model = LeapBotCE.__new__(LeapBotCE)
    nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.torch_dtype = torch.float32
    model.use_cloud_cache = False
    model.mot = SimpleNamespace(num_layers=2, num_heads=2, attn_head_dim=4)

    language = torch.ones(1, 2, 8)
    language_mask = torch.ones(1, 2, dtype=torch.bool)
    model.encode_prompt = lambda _: (language, language_mask)

    def boom(*args, **kwargs):
        raise AssertionError("cloud forward must be skipped when use_cloud_cache=False")

    model._encode_cloud_tensor = boom
    cache = model.encode_cloud(
        torch.zeros(1, 3, 8, 8), prompt="task", cloud_proprio=torch.tensor([[1.0, 2.0]]))

    assert cache.video_seq_len == 0
    assert len(cache.video_cache_k) == len(cache.video_cache_v) == 2
    assert all(t.shape == (1, 0, 8) for t in cache.video_cache_k)
    assert all(t.shape == (1, 0, 8) for t in cache.video_cache_v)
    torch.testing.assert_close(cache.context, language)
    torch.testing.assert_close(cache.context_mask, language_mask)


def test_empty_cloud_cache_attention_covers_prefix_and_action_only():
    # With video_seq_len=0 the denoiser attention mask must cover exactly
    # (edge prefix + action) tokens, and the zero-length cache must cat away.
    model = LeapBotCE.__new__(LeapBotCE)
    nn.Module.__init__(model)
    model.edge_vision = SimpleNamespace(
        mode="select",
        projector=SimpleNamespace(scale_k=torch.ones(2), scale_v=torch.ones(2)))
    model.mot = SimpleNamespace(num_layers=2)

    empty = torch.zeros(1, 0, 8)
    cache = CloudPlanningCache(
        [empty, empty], [empty, empty], torch.ones(1, 2, 8),
        torch.ones(1, 2, dtype=torch.bool), 0, 0)
    captured = {}

    def denoise(noisy, timestep, context, context_mask, cache_k, cache_v,
                action_mask, edge_prefix_k=None, edge_prefix_v=None):
        captured["mask"] = action_mask
        captured["key_shapes"] = [
            torch.cat([k, p], dim=1).shape for k, p in zip(cache_k, edge_prefix_k)]
        return noisy

    model._denoise_action_with_video_cache = denoise
    model._predict_from_cache(
        cache, torch.zeros(1, 5, 8), torch.zeros(1),
        torch.ones(1, 2, 8), torch.ones(1, 2, dtype=torch.bool),
        edge_prefix=torch.ones(1, 3, 8))

    assert captured["mask"].shape == (5, 3 + 5)
    assert all(shape == (1, 3, 8) for shape in captured["key_shapes"])


def test_edge_only_training_skips_cloud_branch_and_stale_loss():
    class TinyTrainScheduler:
        def sample_training_t(self, batch, device, dtype):
            return torch.zeros(batch, dtype=dtype)

        def add_noise(self, action, noise, timestep):
            return action + noise

        def training_target(self, action, noise, timestep):
            return noise

        def training_weight(self, timestep):
            return torch.ones_like(timestep)

    model = LeapBotCE.__new__(LeapBotCE)
    nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.torch_dtype = torch.float32
    model.use_cloud_cache = False
    model.use_cloud_proprio = True
    model.proprio_encoder = None
    model.training_step = 0
    model.stale_loss_lambda_max = 0.5
    model.stale_loss_warmup_steps = 10
    model.mot = SimpleNamespace(num_layers=2, num_heads=2, attn_head_dim=4)
    model.edge_vision = SimpleNamespace(mode="select")
    model.train_action_scheduler = TinyTrainScheduler()

    batch, horizon, dim = 2, 5, 8
    model._prepare_context = lambda sample: (
        torch.ones(batch, 2, dim), torch.ones(batch, 2, dtype=torch.bool))
    model._encode_edge_prefix = lambda views: torch.ones(batch, 3, dim)

    def boom(*args, **kwargs):
        raise AssertionError("cloud branch must not run in edge-only training")

    model._encode_cloud_tensor = boom
    captured = {}

    def predict_from_cache(cache, noisy, timestep, edge_context, edge_mask, edge_prefix=None):
        captured["video_seq_len"] = cache.video_seq_len
        captured["prefix_len"] = int(edge_prefix.shape[1])
        return torch.zeros_like(noisy)

    model._predict_from_cache = predict_from_cache

    sample = {
        "action": torch.zeros(batch, horizon, dim),
        "edge_current_views": torch.zeros(batch, 2, 3, 8, 8),
        "cloud_delay_steps": torch.tensor([20.0, 20.0]),
    }
    loss, metrics = model.training_loss(sample)

    assert torch.isfinite(loss)
    assert captured["video_seq_len"] == 0
    assert captured["prefix_len"] == 3
    assert metrics["loss_stale"] == 0.0
    assert metrics["stale_loss_weight"] == 0.0
    assert metrics["delay_steps"] == pytest.approx(20.0)


def _shell_cache(seq=8, dim=4, layers=2):
    from fastwam.models.wan22.leapbotce import CloudPlanningCache

    k = [torch.arange(seq, dtype=torch.float32).view(1, seq, 1).expand(1, seq, dim).clone()
         for _ in range(layers)]
    v = [(torch.arange(seq, dtype=torch.float32).view(1, seq, 1) + 100.0)
         .expand(1, seq, dim).clone() for _ in range(layers)]
    return CloudPlanningCache(k, v, torch.ones(1, 2, dim),
                              torch.ones(1, 2, dtype=torch.bool), seq, seq)


def test_constant_cache_mode_keeps_layout_but_zeroes_tokens():
    # "Constant cloud tokens" (eval-time): identical shape/sequence length so
    # the attention mask and compute are unchanged; only the content is gone.
    model = LeapBotCE.__new__(LeapBotCE)
    nn.Module.__init__(model)
    model.cloud_cache_mode = "constant"
    cache = _shell_cache()
    out = model._apply_cache_token_ablation(cache)

    assert out.video_seq_len == cache.video_seq_len
    assert out.video_tokens_per_frame == cache.video_tokens_per_frame
    assert [tuple(t.shape) for t in out.video_cache_k] == [tuple(t.shape) for t in cache.video_cache_k]
    assert all(torch.count_nonzero(t) == 0 for t in out.video_cache_k)
    assert all(torch.count_nonzero(t) == 0 for t in out.video_cache_v)
    torch.testing.assert_close(out.context, cache.context)
    torch.testing.assert_close(out.context_mask, cache.context_mask)


def test_shuffled_cache_mode_permutes_tokens_and_keeps_kv_pairing():
    # "Shuffled cloud tokens": the token multiset survives (content statistics
    # identical) but positions are permuted; K and V must share the
    # permutation or the control would test something else entirely.
    model = LeapBotCE.__new__(LeapBotCE)
    nn.Module.__init__(model)
    model.cloud_cache_mode = "shuffled"
    torch.manual_seed(0)
    cache = _shell_cache()
    out = model._apply_cache_token_ablation(cache)

    assert out.video_seq_len == cache.video_seq_len
    for k_in, k_out in zip(cache.video_cache_k, out.video_cache_k):
        torch.testing.assert_close(k_out.sort(dim=1).values, k_in.sort(dim=1).values)
    # v = k + 100 position-wise, so pairing holds iff the permutation matched.
    for k_out, v_out in zip(out.video_cache_k, out.video_cache_v):
        torch.testing.assert_close(v_out, k_out + 100.0)
    assert not torch.equal(out.video_cache_k[0], cache.video_cache_k[0])


def test_full_cache_mode_is_a_no_op():
    model = LeapBotCE.__new__(LeapBotCE)
    nn.Module.__init__(model)
    model.cloud_cache_mode = "full"
    cache = _shell_cache()
    assert model._apply_cache_token_ablation(cache) is cache


def test_single_random_age_training_uses_only_the_aged_branch():
    # "Single-branch random-age" (Table 4): one aged cloud branch per step and
    # a single action loss -- the fresh cloud image must never be encoded.
    class TinyTrainScheduler:
        def sample_training_t(self, batch, device, dtype):
            return torch.zeros(batch, dtype=dtype)

        def add_noise(self, action, noise, timestep):
            return action + noise

        def training_target(self, action, noise, timestep):
            return noise

        def training_weight(self, timestep):
            return torch.ones_like(timestep)

    model = LeapBotCE.__new__(LeapBotCE)
    nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.torch_dtype = torch.float32
    model.use_cloud_cache = True
    model.cloud_cache_mode = "full"
    model.paired_age_mode = "single_random"
    model.use_cloud_proprio = True
    model.proprio_encoder = None
    model.training_step = 0
    model.stale_loss_lambda_max = 0.5
    model.stale_loss_warmup_steps = 10
    model.mot = SimpleNamespace(num_layers=2, num_heads=2, attn_head_dim=4)
    model.edge_vision = SimpleNamespace(mode="select")
    model.train_action_scheduler = TinyTrainScheduler()

    batch, horizon, dim = 2, 5, 8
    model._prepare_context = lambda sample: (
        torch.ones(batch, 2, dim), torch.ones(batch, 2, dtype=torch.bool))
    model._encode_edge_prefix = lambda views: torch.ones(batch, 3, dim)

    def encode_cloud_tensor(image, cloud_context, cloud_context_mask, *_):
        if torch.all(image == 1.0) or torch.any(image == 1.0):
            raise AssertionError("fresh cloud branch must not run in single_random mode")
        from fastwam.models.wan22.leapbotce import CloudPlanningCache
        token = torch.zeros(batch, 4, dim)
        return CloudPlanningCache([token.clone()], [token.clone()],
                                  cloud_context, cloud_context_mask, 4, 4)

    model._encode_cloud_tensor = encode_cloud_tensor
    captured = {}

    def predict_from_cache(cache, noisy, timestep, edge_context, edge_mask, edge_prefix=None):
        captured["video_seq_len"] = cache.video_seq_len
        return torch.ones_like(noisy)

    model._predict_from_cache = predict_from_cache

    sample = {
        "action": torch.zeros(batch, horizon, dim),
        "edge_current_views": torch.zeros(batch, 2, 3, 8, 8),
        "cloud_current_image": torch.ones(batch, 3, 8, 8),
        "cloud_stale_image": torch.full((batch, 3, 8, 8), 2.0),
        "cloud_current_proprio": torch.zeros(batch, 2),
        "cloud_stale_proprio": torch.zeros(batch, 2),
        "cloud_delay_steps": torch.tensor([5.0, 5.0]),
    }
    loss, metrics = model.training_loss(sample)

    assert torch.isfinite(loss)
    assert captured["video_seq_len"] == 4
    assert metrics["loss_fresh"] == 0.0
    assert metrics["stale_loss_weight"] == 1.0
    assert metrics["loss_stale"] > 0.0


def test_encode_cloud_applies_token_mode_end_to_end():
    # encode_cloud is the eval entry point, and the mode is applied inside the
    # REAL _encode_cloud_tensor; fake the three inner primitives so the real
    # method runs end to end rather than stubbing the choke point itself.
    language = torch.ones(1, 2, 8)
    language_mask = torch.ones(1, 2, dtype=torch.bool)
    tokens = (torch.arange(6, dtype=torch.float32).view(1, 6, 1)
              .expand(1, 6, 8).clone())
    for mode in ("constant", "shuffled"):
        model = LeapBotCE.__new__(LeapBotCE)
        nn.Module.__init__(model)
        model.device = torch.device("cpu")
        model.torch_dtype = torch.float32
        model.use_cloud_cache = True
        model.cloud_cache_mode = mode
        model.cloud_kv_adapter = None
        model.cloud_kv_codec = None
        model.mot = SimpleNamespace(
            num_layers=1, num_heads=2, attn_head_dim=4,
            prefill_video_cache_tensor=lambda *a, **k: ([tokens.clone()], [tokens.clone() + 100.0]))
        model.encode_prompt = lambda _: (language, language_mask)
        model._encode_video_latents = lambda frames: torch.zeros(1, 48, 1, 8, 8)
        model.video_expert = SimpleNamespace(
            fuse_vae_embedding_in_latents=False,
            prepare=lambda *a, **k: (tokens, None, None, language, language_mask,
                                     None, None, None, None, 6),
            build_video_to_video_mask=lambda n, per_frame, device: None,
        )
        torch.manual_seed(0)
        cache = model.encode_cloud(torch.zeros(1, 3, 8, 8), prompt="task",
                                   cloud_proprio=torch.tensor([[1.0, 2.0]]))
        assert cache.video_seq_len == 6
        if mode == "constant":
            assert torch.count_nonzero(cache.video_cache_k[0]) == 0
            assert torch.count_nonzero(cache.video_cache_v[0]) == 0
        else:
            torch.testing.assert_close(cache.video_cache_v[0],
                                       cache.video_cache_k[0] + 100.0)
