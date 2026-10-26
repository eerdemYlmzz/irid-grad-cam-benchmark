# Technical Summary: Lesion Localisation in Diabetic Retinopathy

This document reviews the literature that shaped RetinaXAI's design, states what the
project contributes, records the challenges encountered while re-architecting it, and
lists the insights that guided the final pipeline. A note on quantitative results appears
at the end.

---

## 1. Problem framing

Diabetic retinopathy (DR) is graded from colour fundus photographs by the lesions it
produces: microaneurysms (MA), haemorrhages (HE), hard exudates (EX) and soft exudates
(SE). Two related but distinct computer-vision tasks exist:

1. **Image-level grading / classification** - predict a severity label per image.
2. **Pixel-level lesion segmentation** - delineate each lesion.

The original version of this project tried to satisfy a *segmentation* metric (IoU/Dice
against pixel masks) using a *classification* model's saliency maps. The literature below
explains why that is fundamentally limited and what the correct architecture is.

---

## 2. Papers reviewed

### 2.1 Weakly-supervised localisation of DR lesions - Gondal et al. (2017)

The authors localise DR lesions from image-level labels alone using CAM-style evidence
from a CNN. The key reported finding for our purposes is that saliency-based localisation
has low spatial precision: the class-activation evidence identifies broad regions rather
than precise lesion boundaries, and increasing sensitivity to lesion regions introduces
false positives that depress overlap metrics. Grad-CAM-style localisation is reported at
an area-under-PR of roughly 0.18, far below dedicated methods. This is the direct
theoretical justification for why the previous Grad-CAM-only pipeline could not exceed
IoU ~0.1, and why we retain the CAM comparison only as a benchmark, not as the localiser.

### 2.2 Grad-CAM and the CAM family - Selvaraju et al. (2017) and successors

Grad-CAM weights a target layer's activation maps by the gradient of the class score;
Grad-CAM++ refines the weighting for multiple or small objects; Layer-CAM and HiRes-CAM
produce sharper, higher-resolution maps by using element-wise or per-location weighting.
Eigen-CAM differs fundamentally: it takes the first principal component of the target
layer's activations and does not use the class or target at all. This explains, at the
mechanism level, why Eigen-CAM was the worst performer in the original results - its map
is class-agnostic and therefore not lesion-specific. We add Layer-CAM and HiRes-CAM for a
fairer weakly-supervised comparison and keep Eigen-CAM strictly as a negative control.

### 2.3 U-Net++ (nested U-Net) - Zhou et al. (2018)

U-Net++ redesigns the skip connections of U-Net into nested, densely connected pathways
with deep supervision, narrowing the semantic gap between encoder and decoder features.
The reported average IoU gains over U-Net and wide U-Net are on the order of 3-4 points.
For tiny retinal lesions, where fine-grained detail recovery is decisive, this nested
design is a well-motivated default, which is why RetinaXAI's Track A uses U-Net++.

### 2.4 Feature-fusion / residual U-Nets on IDRiD - FFU-Net and DFFR-U-Net

Feature-fusion U-Net variants report strong per-lesion overlap on IDRiD, with mean Dice
around 0.90-0.95 for haemorrhages and hard exudates and near-ceiling scores for the optic
disc. The consistent message across these works is that supervised encoder-decoder
networks, trained on the pixel masks, reach overlap levels that weakly-supervised methods
cannot approach. This sets the target range for Track A.

### 2.5 Attention-gate U-Net with CBAM for microaneurysms - (2024/2025)

This work targets the hardest lesion, microaneurysms, which span roughly 15-60
micrometres. It combines Convolutional Block Attention Modules in the encoder and decoder
with attention gates on the skip connections, and a preprocessing chain of green-channel
extraction, CLAHE, gamma correction, denoising, resizing to 512x512, and 64x64 patch
extraction. It reports Dice 0.865 and IoU 0.758 for microaneurysm segmentation. Two
design cues carried into RetinaXAI: attention/nested skip pathways help small lesions,
and green-channel CLAHE preprocessing materially improves lesion contrast.

### 2.6 DeepLabV3+ for DR lesion segmentation - (2025)

This work applies DeepLabV3+ with per-lesion binary segmentation and LAB-space CLAHE
preprocessing on IDRiD, reporting around 99% pixel accuracy (note: pixel accuracy is a
weak metric under extreme class imbalance and should be read alongside Dice/IoU). Its
main contribution to our design is validating (a) the encoder-decoder-with-ASPP family as
a strong alternative and (b) CLAHE-based contrast enhancement. RetinaXAI exposes
DeepLabV3+ as a one-line configuration switch for direct comparison against U-Net++.

### 2.7 Focal-Tversky loss and Attention U-Net - Abraham and Khan (2019)

The Tversky index generalises Dice by weighting false negatives and false positives
asymmetrically; the focal variant adds an exponent that concentrates learning on hard,
low-overlap examples. On small-lesion segmentation the authors show consistent gains over
Dice and cross-entropy, precisely because lesions are rare relative to background. This is
the basis for RetinaXAI's compound Dice + Focal-Tversky objective, with alpha > beta to
prioritise recall on lesion pixels.

---

## 3. Contribution of this project to the existing picture

RetinaXAI does not claim a new architecture. Its contribution is methodological and
integrative:

1. **A single, comparable benchmark of supervised segmentation against weakly-supervised
   explainability on one shared IoU/Dice protocol and one shared set of IDRiD masks.**
   Most papers report one paradigm or the other; here they are placed on the same axis,
   making the gap explicit and quantified rather than assumed.
2. **A correct evaluation protocol.** Scores aggregate confusion counts across the set
   per lesion before computing IoU/Dice, avoiding the per-image averaging that inflates
   results for the many IDRiD images lacking a given lesion type.
3. **A reproducible, disconnection-resilient training harness** for a small medical
   dataset on free Colab hardware: full-state checkpointing with exact resume, RNG
   restoration, robust mask loading, and memory-aware settings for a T4.
4. **A clear, documented explanation of a negative result** - why Eigen-CAM and, more
   broadly, classifier CAMs cannot serve as lesion segmenters - which turns the original
   project's disappointing numbers into an interpretable finding.

---

## 4. Challenges encountered

- **Task mismatch.** The core difficulty was diagnosing that the low IoU/Dice was
  structural (a classification model scored on a segmentation metric), not a
  hyperparameter problem. The fix required changing the primary model, not tuning it.
- **Extreme class imbalance.** Lesions occupy well under 1% of pixels, so ordinary losses
  collapse to predicting background. This drove the Dice + Focal-Tversky choice.
- **Very small dataset.** IDRiD's segmentation subset has 81 images. This constrains
  model capacity and makes transfer learning, heavy augmentation and strong
  regularisation essential, and it means absolute scores are split-sensitive.
- **Overlapping, multi-scale lesions.** Lesions co-occur and range from a few pixels
  (microaneurysms) to large regions (haemorrhages, exudates), which motivated a
  multi-label sigmoid head and multi-scale-capable decoders.
- **Unreliable compute.** Colab preemption required a resume-safe design so multi-hour
  runs survive disconnection.

---

## 5. Insights and gains

- Supervised segmentation is the only paradigm that reaches the IoU/Dice range clinicians
  and the IDRiD literature expect; explainability heatmaps are for interpretation, not
  measurement.
- Green-channel CLAHE and retina cropping are cheap, evidence-backed preprocessing steps
  that improve lesion contrast before any learning happens.
- Loss design (Focal-Tversky with recall-weighting) is as important as architecture for
  tiny-lesion segmentation.
- Evaluation methodology (dataset-level aggregation) can change reported numbers
  materially and must be stated explicitly for medical benchmarks.

---

## 6. Note on quantitative results

A results-comparison section (this project's trained Track A scores versus the literature
in Section 2, and versus the Track B CAM methods) strictly requires running the training,
which is performed separately on a Colab T4 and is intentionally out of scope for this
setup phase. Those tables are therefore omitted here. The notebook writes them to
`outputs/segmentation_test_results.csv`, `outputs/cam_benchmark_results.csv` and
`outputs/benchmark_comparison.png` once training completes, and this document can be
updated with the measured numbers at that point.

---

## 7. References

1. Gondal et al., "Weakly-supervised localization of diabetic retinopathy lesions in
   retinal fundus images." https://arxiv.org/abs/1706.09634
2. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
   Gradient-based Localization." https://arxiv.org/abs/1610.02391
3. Zhou et al., "UNet++: A Nested U-Net Architecture for Medical Image Segmentation."
   https://arxiv.org/abs/1807.10165
4. "FFU-Net: Feature Fusion U-Net for Lesion Segmentation of Diabetic Retinopathy."
   https://pmc.ncbi.nlm.nih.gov/articles/PMC7801055/
5. "Convolutional block attention gate-based U-Net framework for microaneurysm
   segmentation using retinal fundus images."
   https://pmc.ncbi.nlm.nih.gov/articles/PMC11895248/
6. "Advanced Segmentation of Diabetic Retinopathy Lesions Using DeepLabv3+."
   https://arxiv.org/abs/2504.17306
7. Abraham and Khan, "A Novel Focal Tversky Loss Function with Improved Attention U-Net
   for Lesion Segmentation." https://arxiv.org/abs/1810.07842
