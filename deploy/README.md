---
title: Shift-Aware Skin Lesion Classifier
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
license: cc-by-nc-4.0
short_description: A ViT lesion classifier that refuses unfamiliar photos
---

# Shift-aware skin lesion classifier

**Live demo:** https://skin-lesion-demo.calmplant-b2760678.northcentralus.azurecontainerapps.io/ (Azure Container Apps; the first visit after a quiet spell takes 10 to 20 seconds while it starts). The front matter above also lets this folder run as a Hugging Face Space.

A Vision Transformer (ViT-Tiny, fine-tuned with LoRA) that classifies four lesion types from **dermoscopy** images, plus a shift detector that refuses photos that look unlike its training data, such as ordinary smartphone photos.

**Research and education only. Not medical advice and not a diagnostic tool.** Uploaded images are processed in memory and not stored.

## How it decides

1. **Shift gate.** The squared Mahalanobis distance from the image's CLS embedding to the nearest class mean. Above a threshold that 95% of dermoscopy validation images stay under, the app refuses to guess.
2. **Confidence gate.** For familiar-looking images, the app answers only if the top probability clears a threshold set on validation images.

Across 5 training seeds, the shift gate scores an AUROC of 0.946 between dermoscopy and smartphone photos and catches about 71% of smartphone photos, versus 24% for the model's own confidence at the same false-alarm budget.

## What's in this folder

- `app.py`: the Gradio UI
- `core.py`: preprocessing and both gates, using numpy, Pillow and ONNX Runtime only
- `model/model_int8.onnx`: the int8-quantized model (6.7 MB)
- `model/detector.npz`: class means, the shared precision matrix and both thresholds
- `examples/`: one dermoscopy image and one smartphone photo, credited in `examples/ATTRIBUTION.md`

## Data and licence

Trained on HAM10000 (CC-BY-NC) and evaluated on PAD-UFES-20 (CC-BY), both through the [ISIC Archive](https://www.isic-archive.com). Because the training data is non-commercial, so is this model and demo.

Code, full results and method: [github.com/Janacreates/skin-lesion-shift-peft](https://github.com/Janacreates/skin-lesion-shift-peft)
