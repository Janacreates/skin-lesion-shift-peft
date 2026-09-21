# Fine-tuning under distribution shift: skin lesion classification with LoRA-family methods

> **Status: planning. Nothing has been run yet, and this repository reports no results.** The sections below describe the intended study. The roadmap at the bottom tracks progress and will be updated as work lands.

> **Research and education only.** Not medical advice, and not a diagnostic tool.

## Goal

Fine-tune a pretrained vision model to classify skin lesions using LoRA-family parameter-efficient methods, then test it on images from a **different source than it was trained on**, and study how it fails.

## Why this question

Medical AI models often lose accuracy when they meet data from a different hospital, camera or population. That is called *distribution shift*. A model can also score well by learning shortcuts (a ruler, a sticker, a colour cast, a hospital-specific look) instead of the lesion itself, and then fail elsewhere. Accuracy on a familiar test set hides all of this.

The study asks whether cheaper fine-tuning methods behave differently from full fine-tuning under shift. This is an open question and the answer could go either way. It also asks whether the model's confidence stays honest after the shift, and whether abstaining when unsure catches its mistakes.

## Planned study

**Task.** Classify skin lesion images into diagnostic categories.

**Data (candidates; access and licenses to be confirmed before use).**

| Dataset | What is confirmed so far | Status |
|---|---|---|
| PAD-UFES-20 | 2,298 smartphone clinical photos, 1,373 patients, 6 classes (BCC, melanoma, SCC, actinic keratosis, nevus, seborrheic keratosis); metadata includes age, lesion location and Fitzpatrick skin type; CC BY 4.0 (per its Mendeley Data page) | Page checked. Download size not stated. |
| ISIC 2019 | 25,331 dermoscopic training images across 8 categories (per the challenge page) | Page checked. Sources and license not stated there. |
| HAM10000 | Not yet checked | To confirm: size, classes, license, and how repeated images of one lesion are identified. |

Class definitions differ across datasets (for example, "benign keratosis" versus "seborrheic keratosis"). How to align them is a modelling decision that will be documented.

**Models and methods.**
- A pretrained vision transformer (specific model not yet chosen).
- Baselines: frozen features with a linear classifier, and full fine-tuning.
- Candidate methods: LoRA, DoRA, LoRA+, VeRA, AdaLoRA, LoRA-FA. QLoRA is unlikely to be useful for a model this small.

**Questions.**
1. How large is the drop when a model is tested on a different source?
2. Do parameter-efficient methods hold up better or worse than full fine-tuning under shift?
3. Is the model's confidence still calibrated after the shift?
4. Does abstaining when uncertain remove the errors caused by the shift?
5. Does the model rely on shortcuts such as rulers, markers or colour cast? (Probed by masking and cropping.)
6. Does performance differ by patient group where metadata allows? Group sizes will be small, so error bars will be wide.

**Metrics.** Macro-averaged F1 and per-class recall (the classes are imbalanced), calibration (ECE, negative log-likelihood), risk-coverage curves for abstention, and cost (trainable parameters, memory, training time).

## Rules the study will follow

- Split **by lesion**, not by image, wherever a dataset contains several images of one lesion.
- Check for overlap between sources so that no external test image was seen in training.
- Tune hyperparameters on validation data only, never on the external test set.
- Report mean and standard deviation over multiple seeds.
- Generate tables and figures by script from saved run files. Nothing is typed into this README by hand.

## Open decisions

- [ ] Which shift(s): another dermoscopy source, smartphone photos, or both
- [ ] Which methods to run first (and which to add later)
- [ ] Which pretrained model
- [ ] Time and compute budget

## Roadmap

- [ ] Confirm dataset access and licenses; document the class alignment
- [ ] Data loading with lesion-level splits and overlap checks
- [ ] Baselines: frozen features, full fine-tuning
- [ ] Evaluation harness (macro metrics, calibration, abstention) with tests
- [ ] LoRA fine-tune end to end on a small subset (pipeline check)
- [ ] Remaining methods
- [ ] Shortcut probes and subgroup analysis
- [ ] Multi-seed runs, results table, figures and write-up (including what did not work)

## Limitations to state up front

- A benchmark result is not clinical validity.
- Labels come from different sources with different confirmation standards (biopsy versus consensus).
- Small subgroups will give wide uncertainty.
- Some dermatology datasets carry non-commercial licenses, which will be checked and stated.

## License

MIT (code). Dataset licenses are separate and are listed above as they are confirmed.
