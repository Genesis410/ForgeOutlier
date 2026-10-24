"""CLI unit and functional tests."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from forgeoutlier.cli.commands import main
from forgeoutlier.cli.parser import build_parser
from forgeoutlier.models.loader import compute_file_sha256
from forgeoutlier.models.model import DeepfakeModel
from forgeoutlier.ood.serialization import save_ood_artifact


def test_cli_parser_help() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0
    help_text = parser.format_help()
    assert "extract-embeddings" in help_text
    assert "evaluate" in help_text


def test_cli_inspect_model(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    model_path = tmp_path / "test_model.pt"
    model = DeepfakeModel(feature_dim=576)
    torch.save(model.state_dict(), model_path)

    exit_code = main(["inspect-model", str(model_path)])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "SHA256:" in captured.out
    assert "test_model.pt" in captured.out


def test_cli_fit_ood(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    emb_path = tmp_path / "embs.npy"
    data = np.random.randn(20, 16).astype(np.float32)
    np.save(str(emb_path), data)

    out_path = tmp_path / "profile.npz"
    exit_code = main(
        [
            "fit-ood",
            "--embeddings-path",
            str(emb_path),
            "--output-path",
            str(out_path),
            "--method",
            "cosine",
            "--quantile",
            "0.90",
        ]
    )
    assert exit_code == 0
    assert out_path.exists()
    captured = capsys.readouterr()
    assert "OOD profile fitted successfully" in captured.out


def test_cli_extract_embeddings_and_evaluate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    img_path = img_dir / "test1.jpg"
    img = Image.new("RGB", (224, 224), color=(100, 150, 200))
    img.save(img_path)

    out_npz = tmp_path / "embeddings.npz"
    ret = main(["extract-embeddings", str(img_dir), "--output-path", str(out_npz)])
    assert ret == 0
    assert out_npz.exists()

    data = np.load(out_npz)
    assert "embeddings" in data
    assert data["embeddings"].shape == (1, 576)

    capsys.readouterr()

    manifest_file = tmp_path / "manifest.json"
    manifest_data = [{"path": str(img_path), "label": "real"}]
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    ret_eval = main(["evaluate", "--manifest-path", str(manifest_file), "--json"])
    assert ret_eval == 0
    captured = capsys.readouterr()
    eval_res = json.loads(captured.out)
    assert "total_samples" in eval_res
    assert eval_res["total_samples"] == 1


def test_cli_predict_image_json_and_semantic_exit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # 1. Setup mock pipeline artifacts
    feature_dim = 576
    model = DeepfakeModel(feature_dim=feature_dim)
    checkpoint_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint_path)
    model_sha256 = compute_file_sha256(checkpoint_path)

    manifest_path = tmp_path / "trusted_checkpoints.json"
    manifest_data = {
        "manifest_version": "1.0.0",
        "checkpoints": [{"filename": "model.pt", "sha256": model_sha256}],
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    mu = np.zeros((feature_dim,), dtype=np.float32)
    threshold = 0.50
    ood_path = tmp_path / "ood_profile.npz"
    ood_sha256 = save_ood_artifact(
        file_path=str(ood_path),
        method="cosine",
        mu=mu,
        threshold=threshold,
    )

    img_path = tmp_path / "test.png"
    Image.new("RGB", (224, 224), color=(100, 100, 100)).save(img_path)

    # Predict with --json
    exit_code = main(
        [
            "predict",
            str(img_path),
            "--model-path",
            str(checkpoint_path),
            "--profile-path",
            str(ood_path),
            "--manifest-path",
            str(manifest_path),
            "--ood-sha256",
            str(ood_sha256),
            "--json",
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert parsed["schema_version"] == "1.0.0"
    assert "result" in parsed
    assert "decision" in parsed["result"]
    assert "scores" in parsed

    # Predict with --exit-code-decision
    exit_code_decision = main(
        [
            "predict",
            str(img_path),
            "--model-path",
            str(checkpoint_path),
            "--profile-path",
            str(ood_path),
            "--manifest-path",
            str(manifest_path),
            "--ood-sha256",
            str(ood_sha256),
            "--plain",
            "--exit-code-decision",
        ]
    )
    # Must be either 0 (REAL), 2 (FAKE), or 3 (UNKNOWN)
    assert exit_code_decision in (0, 2, 3)


def test_cli_batch_predict(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    feature_dim = 576
    model = DeepfakeModel(feature_dim=feature_dim)
    checkpoint_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint_path)
    model_sha256 = compute_file_sha256(checkpoint_path)

    manifest_path = tmp_path / "trusted_checkpoints.json"
    manifest_data = {
        "manifest_version": "1.0.0",
        "checkpoints": [{"filename": "model.pt", "sha256": model_sha256}],
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    mu = np.zeros((feature_dim,), dtype=np.float32)
    ood_path = tmp_path / "ood_profile.npz"
    ood_sha256 = save_ood_artifact(
        file_path=str(ood_path),
        method="cosine",
        mu=mu,
        threshold=0.5,
    )

    img1 = tmp_path / "a.png"
    img2 = tmp_path / "b.jpg"
    Image.new("RGB", (224, 224)).save(img1)
    Image.new("RGB", (224, 224)).save(img2)

    exit_code = main(
        [
            "batch-predict",
            str(img1),
            str(img2),
            "--model-path",
            str(checkpoint_path),
            "--profile-path",
            str(ood_path),
            "--manifest-path",
            str(manifest_path),
            "--ood-sha256",
            str(ood_sha256),
            "--json",
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert len(parsed) == 2
