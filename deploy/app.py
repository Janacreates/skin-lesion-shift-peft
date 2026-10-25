"""Gradio demo: a skin lesion classifier that checks whether a photo looks like its training data
before it answers. Runs the int8 ONNX model on CPU; see core.py for the two gates.

    python app.py   # serves on http://localhost:7860 (or $PORT, which cloud hosts set)
"""
from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

from core import DISPLAY_NAMES, SkinClassifier

HERE = Path(__file__).resolve().parent
classifier = SkinClassifier(HERE / "model")

INTRO = """
# Skin lesion classifier that knows when not to guess

This research demo classifies four lesion types (melanoma, basal cell carcinoma, actinic keratosis,
nevus) with a Vision Transformer fine-tuned with LoRA on **dermoscopy** images (taken through a
dermatoscope). On ordinary smartphone photos its accuracy falls from about 80% to about 50%, so
before answering it runs a **shift detector**: if a photo looks unlike the training images, it
says so instead of guessing. If the photo looks familiar but the model is unsure, it says that too.

**Research and education only. Not medical advice and not a diagnostic tool.** Uploaded images
are processed in memory and not stored.
"""

FOOTER = """
How it works, with results across 5 training seeds:
[github.com/Janacreates/skin-lesion-shift-peft](https://github.com/Janacreates/skin-lesion-shift-peft).
Example images: ISIC Archive (HAM10000, CC-BY-NC; PAD-UFES-20, CC-BY); see examples/ATTRIBUTION.md.
"""


def classify(image):
    if image is None:
        return "Upload a photo or pick an example below.", None
    d = classifier.predict(image)
    if d.status == "unfamiliar":
        return (f"### This photo looks different from the images I was trained on\n"
                f"Its shift score is **{d.shift_score:.0f}**, above the limit of {d.shift_threshold:.0f} "
                f"that 95% of dermoscopy validation images stay under, so I won't guess. "
                f"I was trained only on dermoscopy images."), None
    probs = {DISPLAY_NAMES[c]: p for c, p in d.probabilities.items()}
    if d.status == "not_confident":
        return (f"### Not confident\nThe photo looks familiar (shift score {d.shift_score:.0f}), but my top "
                f"probability is only {d.confidence:.0%}, below my {d.confidence_threshold:.0%} threshold, "
                f"so I won't pick a class."), None
    return (f"### Most likely: {DISPLAY_NAMES[d.top_class]} ({d.confidence:.0%})\n"
            f"Shift score {d.shift_score:.0f} (limit {d.shift_threshold:.0f}). "
            f"This is a research model, not a diagnosis."), probs


with gr.Blocks(title="Shift-aware skin lesion classifier") as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="pil", label="Lesion photo")
            button = gr.Button("Classify", variant="primary")
        with gr.Column():
            verdict = gr.Markdown()
            probabilities = gr.Label(num_top_classes=4, label="Class probabilities")
    gr.Examples(examples=[[str(HERE / "examples" / "dermoscopy_example.jpg")],
                          [str(HERE / "examples" / "smartphone_example.jpg")]],
                inputs=image, label="Examples: a dermoscopy image, then a smartphone photo")
    gr.Markdown(FOOTER)
    button.click(classify, inputs=image, outputs=[verdict, probabilities])
    image.change(classify, inputs=image, outputs=[verdict, probabilities])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
