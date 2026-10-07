# Temporal & Multimodal Visual Intelligence

This repository contains the complete submission package for the temporal/multimodal
visual intelligence assessment.

## Repository structure

```text
temporal_multimodal_visual_intelligence/
├── README.md
├── report.pdf
├── requirements.txt
├── src/
├── data/
├── models/
├── embeddings/
├── evaluation/
├── vlm/
├── scripts/
│   ├── temporal-modeling.ipynb
│   ├── temporal_modeling_export.py
│   └── README.md
└── results/
```

## Research question

The experiment investigates whether temporal modeling provides a genuine advantage for
aggressive vs non-aggressive video classification, rather than treating high classification
accuracy alone as evidence of temporal reasoning.

## Pipeline

1. Inspect source videos and sampled frames.
2. Analyze raw pixel changes.
3. Compare pretrained visual representations.
4. Extract normalized 768-dimensional SigLIP embeddings.
5. Construct time-normalized one-second windows with 0.5-second stride.
6. Sample 16 frames per temporal window.
7. Train a single-layer GRU temporal classifier.
8. Compare it with a mean-pooled MLP baseline.
9. Shuffle sequence order at inference time without retraining.
10. Evaluate performance with progressively fewer observations.
11. Run a sequence-level SmolVLM experiment using 8 chronological frames.

## Dataset and split

The supplied technical report describes 11 source videos: 5 aggressive and 6 non-aggressive.
Aggressive source videos are approximately 60 FPS at 1920×1080, while non-aggressive videos
are approximately 25–30 FPS and generally lower resolution.

The temporal experiment produces 169 windows:
- Train: 89 windows (69 aggressive, 20 non-aggressive)
- Validation: 19 windows (4 aggressive, 15 non-aggressive)
- Test: 61 windows (41 aggressive, 20 non-aggressive)

The split is performed at the source-video level to avoid frame leakage. The test set contains
two source videos.

## Models

### SigLIP representation

Encoder:
`google/siglip-base-patch16-224`

Each sampled frame is represented as a normalized 768-dimensional embedding.

### Temporal model

- Input: 16 × 768
- Single-layer GRU
- Hidden size: 128
- Classifier: 128 → 64 → 2
- Optimizer: AdamW
- Learning rate: 1e-3
- Weight decay: 1e-4
- Weighted cross-entropy
- Up to 30 epochs with best validation-loss state restored

### Static baseline

The 16 embeddings are mean-pooled into one 768-dimensional vector and passed to a small MLP.
This preserves the aggregate frame information but removes explicit temporal order.

### VLM

`HuggingFaceTB/SmolVLM-256M-Instruct`

The sequence-level VLM experiment uses 8 uniformly distributed chronological frames from each
16-frame test window and requests an AGGRESSIVE/NON-AGGRESSIVE classification.

## Main results

| Approach / test | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Temporal GRU — ordered | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| MLP | 0.9836 | 0.9762 | 1.0000 | 0.9880 | 1.0000 |
| GRU — shuffled, 10 seeds | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

The key controlled result is that shuffling the 16 embeddings at inference time did not change
the GRU's performance across ten random seeds. Therefore, the 100% ordered-sequence result
should not be interpreted as evidence that the GRU is using temporal order.

## Progressive observation experiment

The trained 16-frame GRU was evaluated using shorter prefixes:

| Available observations | Accuracy | F1 |
|---:|---:|---:|
| 1 | 0.6885 | 0.8119 |
| 2 | 0.9836 | 0.9880 |
| 4 | 1.0000 | 1.0000 |
| 8 | 1.0000 | 1.0000 |
| 16 | 1.0000 | 1.0000 |

This is an observation-availability experiment rather than five independently trained models.

## Interpretation

The static MLP reaches 98.36%, already close to the GRU. More importantly, destroying the
order of the observations leaves the GRU at 100%. The evidence therefore does not establish
a genuine temporal advantage.

The dataset also contains substantial source-level differences in frame rate and resolution,
and the test set contains only two source videos. Static appearance or source-specific
shortcuts are consequently plausible explanations for the high test performance.

The VLM experiment also remained confused on aggressive clips, although it was able to classify
the illustrated non-aggressive case correctly. It is treated as a qualitative result rather
than a definitive benchmark.

## Limitations

- Only 11 source videos.
- Strong frame-rate/resolution differences between classes.
- Validation contains only two source videos.
- Test set contains only two source videos.
- Progressive observation uses prefixes of a model trained on 16-frame sequences.
- Lightweight VLM was constrained by GPU memory.
- High test accuracy should not be interpreted as evidence of a robust violence detector.

## Recommended next experiments

- Use a larger, balanced dataset with more independent source videos.
- Balance camera characteristics, frame rate, resolution, environment, and class.
- Use leave-one-video-out or grouped cross-validation.
- Compare GRU, temporal Transformer, and temporal convolution under identical grouped splits.
- Evaluate original, reversed, and shuffled sequences.
- Introduce controlled temporal corruption.
- Add pose estimation, optical flow, or audio as complementary modalities.
- Use a stronger multi-frame VLM and explicitly ask it to reason about changes between early
  and late observations.

## Reproducibility

The supplied Kaggle notebook contains dataset inspection, embedding extraction,
time-normalized window construction, GRU/MLP training, shuffled-order evaluation,
progressive observation evaluation, and the sequence-level VLM experiment.

The original notebook is preserved at:

`scripts/temporal-modeling.ipynb`

The accompanying technical report is preserved as:

`report.pdf`

## Running

Install dependencies:

```bash
pip install -r requirements.txt
```

Then open:

```text
scripts/temporal-modeling.ipynb
```

The notebook assumes access to the assessment dataset and a GPU-enabled environment.
Dataset files and model checkpoints are not bundled in this ZIP.
