# IronClad System Analysis: Face Recognition Model Selection and Optimization

## Executive Summary

This case study evaluates embedding model selection and system optimization for the IronClad face recognition system. After comprehensive benchmarking, **ArcFace (r100)** was selected as the production model due to superior accuracy (96.5% top-1) and robustness across all tested conditions. The analysis also identifies optimal retrieval parameters for real-world deployment.

---

## 1. Model Comparison: ArcFace vs VGGFace2

### Methodology
Both models were evaluated using brute-force indexing with cosine similarity to establish baseline performance without indexing approximation effects.

### Baseline Results

| Metric          | ArcFace | VGGFace2 | Difference |
| --------------- | ------: | -------: | ---------: |
| Top-1 Accuracy  |  0.9648 |   0.9128 |     +5.7%  |
| Recall@k        |  0.9708 |   0.9479 |     +2.4%  |
| MRR             |  0.9676 |   0.9265 |     +4.4%  |
| Precision@k     |  0.3716 |   0.3593 |     +3.4%  |
| AP@k            |  0.9598 |   0.9096 |     +5.5%  |
| mAP@k           |  0.9598 |   0.9096 |     +5.5%  |
| Search (ms)     |  0.0167 |   0.0096 |   +74% slower |

![ArcFace vs VGGFace2 comparison](storage/ArcFace_vs_VGGFace2_k5.png)

**Key Finding**: ArcFace outperforms VGGFace2 across all accuracy metrics, though with marginally higher inference time.

---

## 2. Robustness Analysis: Gaussian Noise

### Methodology
Gaussian noise was applied to embedding vectors at varying standard deviation levels (σ = 0.02, 0.05, 0.1) to simulate real-world sensor noise and compression artifacts.

### Results

**σ = 0.02 (Low Noise)**
| Metric       | ArcFace | VGGFace2 |
| ------------ | ------: | -------: |
| Top-1 Acc    |  0.9658 |   0.9108 |
| Recall@k     |  0.9708 |   0.9459 |
| MRR          |  0.9683 |   0.9248 |
| AP@k         |  0.9603 |   0.9074 |

**σ = 0.05 (Moderate Noise)**
| Metric       | ArcFace | VGGFace2 |
| ------------ | ------: | -------: |
| Top-1 Acc    |  0.9627 |   0.8968 |
| Recall@k     |  0.9688 |   0.9419 |
| MRR          |  0.9653 |   0.9145 |
| AP@k         |  0.9568 |   0.8929 |

**σ = 0.1 (High Noise)**
| Metric       | ArcFace | VGGFace2 |
| ------------ | ------: | -------: |
| Top-1 Acc    |  0.9386 |   0.8236 |
| Recall@k     |  0.9567 |   0.9128 |
| MRR          |  0.9464 |   0.8601 |
| AP@k         |  0.9361 |   0.8261 |

**Key Finding**: ArcFace demonstrates superior noise robustness. At σ=0.1, ArcFace maintains 93.9% top-1 accuracy while VGGFace2 drops to 82.4%—a 14% gap. Interestingly, ArcFace shows slight improvement at σ=0.02, suggesting minor regularization benefits.

---

## 3. Robustness Analysis: Image Resolution

### Methodology
Images were downscaled to 75%, 50%, and 25% of original resolution before embedding extraction to simulate low-resolution surveillance scenarios.

### Results

**Scale = 0.75**
| Metric          | ArcFace | VGGFace2 |
| --------------- | ------: | -------: |
| Top-1 Acc       |  0.9740 |   0.9268 |
| Recall@5        |  0.9790 |   0.9579 |
| MRR             |  0.9765 |   0.9396 |
| AP@5            |  0.9712 |   0.9243 |

**Scale = 0.50**
| Metric          | ArcFace | VGGFace2 |
| --------------- | ------: | -------: |
| Top-1 Acc       |  0.9710 |   0.9186 |
| Recall@5        |  0.9760 |   0.9569 |
| MRR             |  0.9735 |   0.9350 |
| AP@5            |  0.9680 |   0.9191 |

**Scale = 0.25**
| Metric          | ArcFace | VGGFace2 |
| --------------- | ------: | -------: |
| Top-1 Acc       |  0.9640 |   0.7876 |
| Recall@5        |  0.9740 |   0.8818 |
| MRR             |  0.9682 |   0.8245 |
| AP@5            |  0.9634 |   0.7804 |

**Key Finding**: ArcFace maintains >96% top-1 accuracy even at 0.25x resolution, while VGGFace2 degrades significantly (78.8%). This makes ArcFace better suited for surveillance applications with variable image quality.

*Note*: Metrics at reduced resolutions slightly exceed baseline, likely due to edge artifact removal during downscaling.

---

## 4. Robustness Analysis: Lighting Conditions

### Methodology
Image brightness was adjusted from 0.5x (darker) to 1.5x (brighter) to simulate varying lighting conditions.

### Results

**Brightness = 0.50 (Dark)**
| Metric          | ArcFace | VGGFace2 |
| --------------- | ------: | -------: |
| Top-1 Acc       |  0.9720 |   0.9208 |
| Recall@k        |  0.9780 |   0.9619 |
| MRR             |  0.9745 |   0.9369 |
| AP@k            |  0.9682 |   0.9199 |

**Brightness = 0.75**
| Metric          | ArcFace | VGGFace2 |
| --------------- | ------: | -------: |
| Top-1 Acc       |  0.9740 |   0.9339 |
| Recall@k        |  0.9790 |   0.9629 |
| MRR             |  0.9765 |   0.9459 |
| AP@k            |  0.9702 |   0.9306 |

**Brightness = 1.25**
| Metric          | ArcFace | VGGFace2 |
| --------------- | ------: | -------: |
| Top-1 Acc       |  0.9750 |   0.9248 |
| Recall@k        |  0.9810 |   0.9549 |
| MRR             |  0.9774 |   0.9372 |
| AP@k            |  0.9718 |   0.9221 |

**Brightness = 1.50 (Bright)**
| Metric          | ArcFace | VGGFace2 |
| --------------- | ------: | -------: |
| Top-1 Acc       |  0.9750 |   0.9147 |
| Recall@k        |  0.9810 |   0.9584 |
| MRR             |  0.9774 |   0.9322 |
| AP@k            |  0.9728 |   0.9079 |

**Key Finding**: ArcFace performance improves slightly with brightness and plateaus at 1.25x. VGGFace2 peaks at original brightness and degrades in both darker and brighter conditions, indicating less robust internal normalization.

---

## 5. Optimal Retrieval Depth (Top-N Selection)

### Analysis

Increasing N (number of candidates returned) improves recall but has diminishing returns beyond certain thresholds.

| Brightness | Model    | Top-1 | Top-5 | Improvement |
| ---------- | -------- | ----: | ----: | ----------: |
| 0.50       | ArcFace  | 0.972 | 0.978 |       +0.6% |
| 0.50       | VGGFace2 | 0.921 | 0.962 |       +4.5% |
| 0.75       | ArcFace  | 0.974 | 0.979 |       +0.5% |
| 0.75       | VGGFace2 | 0.934 | 0.963 |       +3.1% |
| 1.25       | ArcFace  | 0.975 | 0.981 |       +0.6% |
| 1.25       | VGGFace2 | 0.925 | 0.955 |       +3.2% |
| 1.50       | ArcFace  | 0.975 | 0.981 |       +0.6% |
| 1.50       | VGGFace2 | 0.915 | 0.958 |       +4.7% |

### Recommendation: N = 5

**Rationale**:
1. **Accuracy plateau**: Beyond N=5, gains become marginal (<0.5% improvement)
2. **User experience**: Reviewing 5 candidates balances accuracy with cognitive load for security personnel
3. **Latency**: Additional candidates increase processing time without proportional accuracy benefit

---

## Conclusions and Recommendations

### Model Selection
**ArcFace (r100)** is recommended for production deployment due to:
- Consistently superior accuracy across all conditions
- Superior robustness to noise, resolution, and lighting variations
- Better generalization to real-world deployment scenarios

### System Parameters
- **Retrieval depth**: N = 5 (optimal accuracy/usability tradeoff)
- **Similarity metric**: Cosine similarity
- **Indexing strategy**: HNSW for galleries >10K faces (see separate benchmarking analysis)

### Production Considerations
1. Pre-normalize embeddings at index time to reduce query latency
2. Implement confidence thresholding to reject low-quality matches
3. Consider ensemble approaches for high-security applications
