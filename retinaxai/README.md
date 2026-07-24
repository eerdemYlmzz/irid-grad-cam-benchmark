# RetinaXAI: Diabetic Retinopathy Lesion Segmentation and Explainability Benchmark

RetinaXAI localises diabetic-retinopathy (DR) lesions in colour fundus photographs and
measures how well the localisation matches expert pixel-level annotations, using
Intersection-over-Union (IoU) and the Dice coefficient. The project is built around the
IDRiD dataset and is delivered as a single, self-contained Colab notebook that runs on a
free T4 GPU.

The repository contains two tracks that share the same evaluation masks and the same
IoU/Dice protocol, so their results sit in one comparable table:

- **Track A - Supervised lesion segmentation.** A U-Net++ encoder-decoder with a
  pretrained EfficientNet encoder is trained directly on IDRiD lesion masks. This is the
  model that produces high IoU/Dice and is the recommended localiser.
- **Track B - Explainability benchmark.** A DR-severity classifier is probed with the
  Grad-CAM family (Grad-CAM, Grad-CAM++, Layer-CAM, HiRes-CAM, Eigen-CAM) and the
  resulting heatmaps are scored against the same masks. This quantifies the gap between
  weakly-supervised saliency and a fully supervised segmenter.

---

## 1. Motivation: why the previous version scored so low

The earlier pipeline trained a classifier on APTOS severity labels, produced Grad-CAM /
Eigen-CAM heatmaps, and compared them to IDRiD lesion masks. The measured overlap was
very low (IoU roughly 0.005-0.10, Dice 0.01-0.18, Eigen-CAM worst of all).

This is not a bug that thresholding can fix. It is the documented ceiling of
weakly-supervised localisation:

- **Class Activation Maps show discriminative regions, not lesion extent.** A classifier
  only needs to find enough evidence to pick a severity grade. A small cluster of
  microaneurysms can decide the label, so the heatmap forms one coarse blob and ignores
  the remaining lesions that the mask still counts.
- **Eigen-CAM does not use the class at all.** It returns the first principal component
  of the target-layer activations, so its map is not lesion-specific. A poor Eigen-CAM
  overlap is the expected behaviour, not a tuning failure. We keep it only as a negative
  control.
- **Lesions are tiny.** Microaneurysms span roughly 15-60 micrometres, a handful of
  pixels. A heatmap upsampled from a coarse feature map cannot resolve them.

The rational fix, consistent with the IDRiD literature, is to train a dedicated
segmentation network on the pixel-level masks. Track A does exactly that; Track B keeps
the explainability comparison as an honest, quantified baseline rather than as the
primary localiser.

---

## 2. Method

### 2.1 Data

- **IDRiD segmentation subset** (81 fundus images, 54 train / 27 test as split by the
  organisers) with four expert lesion masks per image: microaneurysms (MA), haemorrhages
  (HE), hard exudates (EX), soft exudates (SE). Track A trains and is evaluated here.
- **APTOS-2019** severity-graded fundus images. Only Track B uses these, to train the
  classifier that the CAM methods explain.

### 2.2 Preprocessing

- **Retina crop.** The circular field of view is detected and cropped away from the black
  border so lesions occupy more of the frame. The same crop box is applied to every mask
  to keep pixels aligned.
- **CLAHE on the green channel.** The green channel gives the strongest contrast between
  lesions and background; Contrast Limited Adaptive Histogram Equalisation locally
  amplifies faint microaneurysms and exudates without saturating bright regions.

### 2.3 Segmentation model (Track A)

- **Architecture:** U-Net++ (nested, densely connected skip pathways) with an
  EfficientNet-B4 encoder pretrained on ImageNet, via `segmentation-models-pytorch`.
  A DeepLabV3+ variant is available through a single config switch.
- **Output:** four sigmoid channels (multi-label), because lesions overlap and a pixel
  can belong to more than one lesion type.
- **Loss:** a weighted sum of Dice loss and Focal-Tversky loss. Focal-Tversky upweights
  false negatives (missing a lesion is worse than a false alarm) and focuses the gradient
  on the rare, hard lesion pixels, which is the standard remedy for the extreme
  foreground/background imbalance in retinal lesions.
- **Training:** AdamW, OneCycle learning-rate schedule, automatic mixed precision (AMP),
  gradient accumulation for an effective batch of eight, heavy geometric and mild
  photometric augmentation to compensate for the small training set, and early stopping
  on validation mean-Dice.

### 2.4 Explainability benchmark (Track B)

An EfficientNet-B0 classifier trained on APTOS is probed at a mid-level layer (finer
spatial resolution than the final block) with Grad-CAM, Grad-CAM++, Layer-CAM, HiRes-CAM
and Eigen-CAM. Each heatmap is thresholded per image with Otsu and scored against the
combined (union of all four) lesion mask, exactly the target Track A also reports.

### 2.5 Classical baseline

A compact classical pipeline (CLAHE, morphological white top-hat, Otsu threshold) for
hard exudates is included as a reference floor, so the added value of the deep model over
plain image processing is explicit.

### 2.6 Evaluation

All scores aggregate the confusion counts (TP/FP/FN) across the whole set per lesion and
compute IoU/Dice once (dataset-level), rather than averaging per-image scores. On IDRiD,
most images lack at least one lesion type, and per-image averaging silently inflates the
numbers for the absent lesions. Dataset-level aggregation matches how the IDRiD
leaderboard reports and is the honest figure.

---

## 3. Repository layout

```
retinaxai/
  notebooks/
    RetinaXAI_Pipeline.ipynb    End-to-end pipeline (both tracks)
  outputs/                      Figures and CSV result tables (written by the notebook)
  docs/
    TECHNICAL_SUMMARY.md        Literature review, contribution, challenges, insights
  requirements.txt
  README.md
```

Expected data layout on Google Drive (edit `CFG.BASE` in the notebook if yours differs):

```
retinaxai/
  data/
    idrid/
      images/                 IDRiD_XX.jpg
      masks/
        Microaneurysms/       IDRiD_XX_MA.tif
        Haemorrhages/         IDRiD_XX_HE.tif
        HardExudates/         IDRiD_XX_EX.tif
        SoftExudates/         IDRiD_XX_SE.tif
    aptos2019/                (Track B only)
      subset_large.csv
      subset_images_large/
  checkpoints/                Written by the notebook (resume-safe)
  outputs/                    Written by the notebook
```

---

## 4. How to run

1. Open `notebooks/RetinaXAI_Pipeline.ipynb` in Google Colab and select a T4 GPU runtime.
2. Run the setup cells (Drive mount, installs, config). Adjust `CFG.BASE` to your Drive
   path if needed.
3. Run Track A to train the segmentation model and produce the IoU/Dice tables.
4. Optionally run Track B for the explainability comparison chart.

The notebook is designed for a single session but is robust to interruption (see below).

---

## 5. Robustness and engineering details

- **Disconnection-safe checkpointing.** Every epoch writes `checkpoints/last.pt` with the
  complete training state: model, optimiser, scheduler, AMP scaler, epoch counter,
  training history and all RNG states. `checkpoints/best.pt` is written whenever
  validation mean-Dice improves. If the Colab VM is preempted, simply re-run the training
  cell: it detects `last.pt` and resumes at the exact epoch it stopped, so at most one
  epoch of work is lost.
- **Deterministic and reproducible.** All RNGs are seeded and restored on resume.
- **Robust data loading.** Missing lesion masks are treated as all-zero channels rather
  than errors (not every image contains every lesion). A corrupt or out-of-memory batch
  is caught and skipped instead of killing a multi-hour run.
- **Memory-aware for T4.** Mixed precision plus gradient accumulation keep an
  EfficientNet-B4 U-Net++ at 512x512 inside a 15 GB T4. If you hit an out-of-memory
  error, reduce `IMG_SIZE` to 384, drop the encoder to `efficientnet-b2`, or lower
  `BATCH_SIZE` and raise `GRAD_ACCUM` to keep the effective batch constant.
- **Portable.** The notebook detects whether it is on Colab and skips the Drive mount
  when run locally.

---

## 6. Outputs

The notebook writes to `outputs/`:

- `segmentation_test_results.csv` - per-lesion and combined IoU/Dice on the IDRiD test
  split (Track A).
- `cam_benchmark_results.csv` - per-image IoU/Dice for every CAM method (Track B).
- `benchmark_comparison.png` - supervised segmentation vs. every CAM method on one axis.
- `seg_training_curves.png`, `segmentation_qualitative.png` - training curves and
  qualitative overlays.

---

## 7. Configuration knobs

All hyperparameters live in the `CFG` class in the notebook. The most useful ones:

| Setting | Default | Purpose |
| --- | --- | --- |
| `IMG_SIZE` | 512 | Input resolution; lower it to save memory. |
| `ENCODER` | `efficientnet-b4` | Encoder backbone; `b2`/`b3` are lighter. |
| `ARCH` | `unetplusplus` | `unetplusplus` or `deeplabv3plus`. |
| `BATCH_SIZE` / `GRAD_ACCUM` | 4 / 2 | Effective batch = product. |
| `TVERSKY_ALPHA` / `BETA` | 0.7 / 0.3 | Recall vs. precision emphasis on lesions. |
| `THRESHOLD` | 0.5 | Probability-to-mask cut (evaluation sweeps this too). |
| `USE_CLAHE` | True | Toggle green-channel CLAHE preprocessing. |

---

## 8. Reference results from the literature

These are the numbers reported by other work on IDRiD; they are the target range for
Track A once trained. They are not this project's results (training is run separately).

- Dedicated attention U-Net for microaneurysms: Dice ~0.865, IoU ~0.758.
- U-Net variants for haemorrhages and hard exudates: Dice ~0.90-0.95.
- Grad-CAM lesion localisation (weakly supervised): area-under-PR ~0.18, far below the
  supervised segmenters, which is consistent with the low overlap Track B measures.

See `docs/TECHNICAL_SUMMARY.md` for the full literature review and citations.

---

## 9. Notes and limitations

- IDRiD's segmentation subset is small (81 images). Transfer learning, heavy
  augmentation and strong regularisation mitigate this, but absolute numbers depend on
  the train/validation split.
- Soft exudates are the rarest lesion and typically the hardest channel; expect its Dice
  to trail the others.
- Track B is an interpretability study, not a segmentation method. Its low overlap is the
  expected and intended finding, not a defect.
