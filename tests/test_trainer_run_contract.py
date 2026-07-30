import pytest

from fastwam.trainer import Wan22Trainer


def test_resume_run_contract_accepts_exact_metadata(monkeypatch):
    monkeypatch.setenv("LEAPBOT_RUN_CONTRACT_SHA256", "abc123")
    monkeypatch.setenv("LEAPBOT_CODE_COMMIT", "deadbeef")
    Wan22Trainer._validate_resume_run_contract(
        {"run_contract_sha256": "abc123", "code_commit": "deadbeef"}
    )


@pytest.mark.parametrize(
    "payload,match",
    [
        ({"code_commit": "deadbeef"}, "run_contract_sha256"),
        (
            {"run_contract_sha256": "different", "code_commit": "deadbeef"},
            "run_contract_sha256",
        ),
        (
            {"run_contract_sha256": "abc123", "code_commit": "different"},
            "code_commit",
        ),
    ],
)
def test_resume_run_contract_rejects_missing_or_changed_metadata(
    monkeypatch, payload, match
):
    monkeypatch.setenv("LEAPBOT_RUN_CONTRACT_SHA256", "abc123")
    monkeypatch.setenv("LEAPBOT_CODE_COMMIT", "deadbeef")
    with pytest.raises(ValueError, match=match):
        Wan22Trainer._validate_resume_run_contract(payload)


def test_generic_fastwam_resume_has_no_contract_requirement(monkeypatch):
    monkeypatch.delenv("LEAPBOT_RUN_CONTRACT_SHA256", raising=False)
    monkeypatch.delenv("LEAPBOT_CODE_COMMIT", raising=False)
    Wan22Trainer._validate_resume_run_contract({})
