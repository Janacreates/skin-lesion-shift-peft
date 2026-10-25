import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor

from deploy.core import SkinClassifier, decide, preprocess, save_detector
from skinshift.export import Deployable, export_onnx, quantize_int8
from skinshift.model import CHECKPOINT


def test_app_preprocessing_matches_the_hugging_face_processor():
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 256, size=(300, 260, 3), dtype=np.uint8))
    expected = ViTImageProcessor.from_pretrained(CHECKPOINT)(img, return_tensors="np")["pixel_values"]
    np.testing.assert_allclose(preprocess(img), expected, atol=1e-6)


def test_decide_refuses_unfamiliar_images_then_unsure_ones_then_predicts():
    means = np.array([[0.0, 0.0], [10.0, 0.0]])
    precision = np.eye(2)
    args = dict(means=means, precision=precision, shift_threshold=4.0, confidence_threshold=0.6,
                class_names=["a", "b"])
    confident, unsure = np.array([5.0, 0.0]), np.array([0.1, 0.0])
    assert decide(confident, np.array([50.0, 50.0]), **args).status == "unfamiliar"
    assert decide(unsure, np.array([0.5, 0.0]), **args).status == "not_confident"
    d = decide(confident, np.array([0.5, 0.0]), **args)
    assert d.status == "prediction" and d.top_class == "a"
    assert abs(sum(d.probabilities.values()) - 1.0) < 1e-6


def _tiny_vit():
    """A small random ViT with the real model's input size, so the export path runs quickly in CI."""
    torch.manual_seed(0)
    cfg = ViTConfig(image_size=224, patch_size=16, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                    intermediate_size=64, num_labels=4)
    return ViTForImageClassification(cfg).eval()


def test_onnx_export_matches_pytorch_and_runs_in_the_app(tmp_path):
    model = _tiny_vit()
    fp32, int8 = tmp_path / "model_fp32.onnx", tmp_path / "model_int8.onnx"
    export_onnx(model, fp32)
    quantize_int8(fp32, int8)

    px = torch.randn(3, 3, 224, 224)
    with torch.no_grad():
        ref_logits, ref_feats = (t.numpy() for t in Deployable(model)(px))
    sess = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"])
    logits, feats = sess.run(["logits", "features"], {"pixel_values": px.numpy()})  # batch of 3: dynamic axis
    np.testing.assert_allclose(logits, ref_logits, atol=1e-4)
    np.testing.assert_allclose(feats, ref_feats, atol=1e-4)

    # The quantized model plus a saved detector load through the same class the app uses.
    save_detector(tmp_path / "detector.npz", means=ref_feats[:2], precision=np.eye(32), shift_threshold=1e9,
                  confidence_threshold=0.0, class_names=["melanoma", "bcc", "actinic_keratosis", "nevus"])
    clf = SkinClassifier(tmp_path)
    d = clf.predict(Image.new("RGB", (240, 200), (180, 120, 110)))
    assert d.status == "prediction" and set(d.probabilities) == {"melanoma", "bcc", "actinic_keratosis", "nevus"}
