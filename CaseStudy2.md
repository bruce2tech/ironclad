
#### Task 1: Using Brute Force, compare the performance of different distance measures (e.g., euclidean, cosine similarity) using mean Average Precision (mAP) and Recall on the CASIA-WebFace and VGGFace2 datasets. Contrast these results with Mean Reciprocal Rank (MRR). Suggest the distance measure to implement for the system for Task 2-5. 


Tables 1-2 and figures 1-4 show VGGFace2 model outperforms Casia-Webface. The difference in performance could be attributed towards the model choices, not the distance measurements.  For both  models mAP and Recall the euclidean cosine and dot-product metric values were identical, while minkowski was slightly lower. The MRR values were the same across all distance measurements with respect to the model. 

#### VGGFace2 

| Metric | euclidean | cosine | dot_product | minkowski_p1 |
|---|---|---|---|---|
| mAP@K | 0.4761 | 0.4761 | 0.4761 | 0.4752 |
| Recall@K | 0.6647 | 0.6647 | 0.6647 | 0.6597 |
| MRR (baseline=Euclidean) | 0.5658 | — | — | — |

Table 1: List of the metric values for each euclidean, cosine, dot-product and, minkowski distance measurements. MRR is the same across all distance measurements

![Alt text](storage/mi_eval/compare_distances_charts/vggface2_mean_average_precision_at_k.png)

Figure 1: A side by side visual comparison of mean Average precision for VGGFace2 model implemented with each distance measurement.

![Alt text](storage/mi_eval/compare_distances_charts/vggface2_recall_at_k.png)

Figure 2: A side by side visual comparison of Recall@K for VGGFace2 model implemented with each distance measurement.

#### Casia-Webface

| Metric | euclidean | cosine | dot_product | minkowski_p1 |
|---|---|---|---|---|
| mAP@K | 0.0798 | 0.0798 | 0.0798 | 0.0771 |
| Recall@K | 0.1632 | 0.1632 | 0.1632 | 0.1642 |
| MRR (baseline=Euclidean) | 0.1177 | — | — | — |

Table 2: List of the metric values for each euclidean, cosine, dot-product and, minkowski distance measurements. MRR is the same across all distance measurements

![Alt text](storage/mi_eval/compare_distances_charts/casia-webface_mean_average_precision_at_k.png)

Figure 3: A side by side visual comparison of mean Average Precision for Casia-Webface model implemented with each distance measurement.

![Alt text](storage/mi_eval/compare_distances_charts/casia-webface_recall_at_k.png)

Figures 4: A side by side visual comparison of Recall@K for Casia-Webface model implemented with each distance measurement.

For tasks 2-5, the analysis will use cosine distance measurement because it is naturally scale invariant when averaging multiple images per identity and IP and L2 normalization is consistent across bruteforce HNSW and LSH.


#### Task 2: In the dataset, each individual have varying number of images (m) in the catalog/gallery. Investigate how retrieval performance on CASIA-WebFace varies as the number of images per individual in the gallery/catalog increases (m = 1, 2, 3, ...). Suggest the optimal m, supported by your findings, and discuss dataset-specific factors that may influence your conclusion. 


Top-1 accuracy climbs to a peak at m=3 (0.2033) and then drops slightly at m=4–5.

Recall@5 increases monotonically from 0.2276 → 0.4797 as m grows (more gallery evidence makes it likelier the correct identity appears somewhere in top-5).

MRR / mAP@K rise steadily and plateau by m=4–5 (0.2869 → 0.2896).
(Because the gallery has one centroid per identity, there’s at most one relevant item—so AP@K equals MRR, which the numbers reflect.)

Precision@5 also increases, but more gradually (0.0455 → 0.0959).

#### CASIA-WebFace — Cosine (centroids) — Metrics vs m

| m | num_identities | top1_acc | recall_at_k | mrr | precision_at_k | average_precision_at_k | mean_average_precision_at_k |
|---|---|---|---|---|---|---|---|
| 1 | 123 | 0.0894 | 0.2276 | 0.1329 | 0.0455 | 0.1329 | 0.1329 |
| 2 | 123 | 0.1057 | 0.3496 | 0.1924 | 0.0699 | 0.1924 | 0.1924 |
| 3 | 123 | 0.2033 | 0.3740 | 0.2649 | 0.0748 | 0.2649 | 0.2649 |
| 4 | 123 | 0.1951 | 0.4390 | 0.2869 | 0.0878 | 0.2869 | 0.2869 |
| 5 | 123 | 0.1870 | 0.4797 | 0.2896 | 0.0959 | 0.2896 | 0.2896 |

Table 3: Displays how metric values change as m increases


If the goal is identification with highest Top-1, pick m = 3 (peak Top-1 = 0.2033).

However, if retrieval is more important, prefer m = 4–5, which gives higher Recall@5 (0.4390–0.4797) and slightly better MRR/mAP with diminishing returns from 4→5 (0.2869 → 0.2896).

For a balanced, practical default choose m = 4.

For m=3: Top-1 −0.0082, but Recall@5 +0.0650 and MRR/mAP +0.0220.

There is a noticeable bump in ranking quality and recall for a tiny Top-1 trade-off.

Dataset-specific factors (CASIA-WebFace) that push toward this pattern:
Less intra-class coverage per identity (vs. VGGFace2) and older, noisier labels → more chance that additional images include outliers or very different conditions, which pulls the centroid.


#### Task 3: Show the impact of the number of images (m= 1, 2, 3, ...) per individual in the catalog/gallery on the retrieval performance on VGGFace2. Suggest the optimal m, supported by your findings, and discuss dataset-specific factors that may influence your conclusion. Finally, based on your findings, suggest the embedding model to implement for your system.


* From m = 1-4 there are steady gains:
* Top-1: 0.5528 - 0.7886 (+0.236)
* Recall@K: 0.7561 - 0.9187 (+0.163)
* MRR/mAP: 0.6394 - 0.8402 (+0.201)
* From m = 4 - 5 the metrics plateau:
* Top-1: no change (0.7886 - 0.7886)
* Recall@K: no change (0.9187 - 0.9187)
* MRR/mAP: tiny +0.0005

#### VGGFace2 — Cosine (centroids) — Metrics vs m

| m | num_identities | top1_acc | recall_at_k | mrr | precision_at_k | average_precision_at_k | mean_average_precision_at_k |
|---|---|---|---|---|---|---|---|
| 1 | 123 | 0.5528 | 0.7561 | 0.6394 | 0.1512 | 0.6394 | 0.6394 |
| 2 | 123 | 0.6585 | 0.8374 | 0.7351 | 0.1675 | 0.7351 | 0.7351 |
| 3 | 123 | 0.7236 | 0.8862 | 0.7923 | 0.1772 | 0.7923 | 0.7923 |
| 4 | 123 | 0.7886 | 0.9187 | 0.8402 | 0.1837 | 0.8402 | 0.8402 |
| 5 | 123 | 0.7886 | 0.9187 | 0.8407 | 0.1837 | 0.8407 | 0.8407 |

Table 4: Displays how metric values change as m increases


The clear “knee” is m = 4. It captures most of the intra-identity variation by 4 images; the 5th adds negligible accuracy but still costs one more forward pass per identity when (re)building the gallery.

Implement FaceNet (InceptionResnetV1) pretrained on VGGFace2.
The earlier CASIA-WebFace runs were far lower across all metrics; VGGFace2 is clearly superior in this setup. Given this fact, the remaining analysis will focus on optimizing the implementation with VGGFace2 model.


#### Task 4: Measure the impact of selecting Brute Force vs HNSW on the retrieval performance. Estimate Brute Force's and HNSW's performance on a billion images (as per the requirements).


Accuracy: Identical across Top-1, Recall@K, MRR, AP@K, mAP@K.
This is expected with such a tiny gallery and efSearch=64—HNSW essentially returns the same top-k as exact search.

Latency: HNSW is faster on this run.
Flat = 0.0681 ms/query, HNSW = 0.0435 ms/query → HNSW is ~36% faster.
(Note: at this scale, absolute times are dominated by fixed overheads; the ratio is more meaningful than the raw µs.)

Build & memory: Nearly the same; HNSW shows a small overhead.
Build: 0.000207 s (Flat) vs 0.000230 s (HNSW)
Index mem: 0.092 MB (Flat) vs 0.104 MB (HNSW)

#### VGGFace2: BruteForce vs HNSW Metrics

| index | top1\_acc | recall\_at\_k | mrr      | precision\_at\_k | average\_precision\_at\_k | mean\_average\_precision\_at\_k | avg\_search\_ms | build\_time\_s | index\_mem\_mb |
| ----- | --------- | ------------- | -------- | ---------------- | ------------------------- | ------------------------------- | --------------- | -------------- | -------------- |
| FLAT  | 0.444444  | 0.777778      | 0.583333 | 0.444444         | 0.578241                  | 0.578241                        | 0.068116        | 0.000207       | 0.091840       |
| HNSW  | 0.444444  | 0.777778      | 0.583333 | 0.444444         | 0.578241                  | 0.578241                        | 0.043537        | 0.000230       | 0.104025       |

Table 5: Displays side by side comparison of accuracy metrics of bruteforce indexing vs HNSW indexing

![Alt text](storage/mi_eval/bf_vs_hnsw_vggface2_plots/accuracy_flat_vs_hnsw.png)

Figure 5: Displays side by side bar chart comparison of accuracy metrics of bruteforce indexing vs HNSW indexing

![Alt text](storage/mi_eval/bf_vs_hnsw_vggface2_plots/time_flat_vs_hnsw.png)

Figure 6: Displays side by side comparison of search time metrics of bruteforce indexing vs HNSW indexing


#### Metric Estimates for Large Dataset Performance
| Item                                      |             Value | Units     | Notes                 |
| ----------------------------------------- | ----------------: | --------- | --------------------- |
| Per-query search time (Flat)              |           37.9096 | s/query   | Linear scan estimate  |
| Per-query search time (HNSW)              |          0.000122 | s/query   | ≈0.122 ms/query       |
| Assumption: flat\_linear\_per\_vector\_ms |          3.79e-05 | ms/vector | ≈3.79e-08 s/vector    |
| Assumption: hnsw\_log\_scaling\_factor    |            2.6825 | —         | Cost \~ efSearch·logN |
| Memory — vectors                          | 2,048,000,000,000 | bytes     | ≈1.9 TB               |
| Memory — HNSW graph (low)                 |   128,000,000,000 | bytes     | ≈119.2 GB             |
| Memory — HNSW graph (high)                |   256,000,000,000 | bytes     | ≈238.4 GB             |

Table 6: Presents an estimate of metrics for Brute Force and HNSW indexing strategies applied to large datasets.

What the estimates imply:

Flat @ 1B: likely tens of seconds per query (not minutes) on a single node with vectors in RAM; still impractical without heavy batching/sharding/GPUs/compression because ~2 TB per query must read

HNSW @ 1B: sub-millisecond per query if efSearch is fixed and everything is in memory—but recall typically drops as N grows, so often efSearch is raised (trading more latency for accuracy). Also, need ~1.9 TB (vectors) + ~120–240 GB (graph) in RAM (or shard/compress).
To keep recall as N grows, efSearch is often increased, (latency grows ~linearly with efSearch).

You must store ~1.9 TB of vectors plus ~120–240 GB for the graph; most single machines can’t hold this, and may need sharding and/or compression (e.g., PQ/OPQ, IVF+PQ, or HNSW over compressed centroids).

Real performance is bounded by memory bandwidth and system architecture (NUMA, cache behavior, disk vs RAM vs GPU HBM, etc.).


#### Task 5: Measure the impact of selecting HNSW vs LSH on the retrieval performance. Estimate LSH's performance on a large amount of images (as per the requirements). Discuss situations where LSH or HSNW might be preferred, even at the cost of some accuracy.

Accuracy

Identical across all metrics (Top-1, Recall@K, MRR, AP@K, mAP@K).
With such a tiny gallery (G=47) and efSearch=64, HNSW is effectively exact. The LSH settings (256 bits + rerank=200) also recover the same top-k here.

Latency (per query)
* HNSW: 0.010671 ms
* LSH: 0.041968 ms
* HNSW is ~3.9× faster (0.041968 / 0.010671 ≈ 3.93).

Build time
* HNSW: 0.000251 s
* LSH: 0.000087 s
* LSH builds ~2.9× faster (still negligible at this scale).

Index memory (just the index, not vectors)
* HNSW: 0.104 MB
* LSH: 0.502 MB
* LSH index blob is ~4.8× larger at this size (but see scale notes below).

### Metrics & Timings (HNSW vs LSH)

| index | top1_acc | recall_at_k | mrr | precision_at_k | average_precision_at_k | mean_average_precision_at_k | avg_search_ms | build_time_s | index_mem_mb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HNSW | 0.444444 | 0.777778 | 0.583333 | 0.444444 | 0.578241 | 0.578241 | 0.010671 | 0.000251 | 0.104025 |
| LSH | 0.444444 | 0.777778 | 0.583333 | 0.444444 | 0.578241 | 0.578241 | 0.041968 | 0.000087 | 0.501523 |

Table 7: Accuracy and timing metrics for HNSW and LSH


![Alt text](results_hnsw_lsh/accuracy_hnsw_vs_lsh.png)
Figure 7: Side by side comparison of HNSW and LSH accuracy metrics

![Alt text](results_hnsw_lsh/time_hnsw_vs_lsh.png)
Figure 8: Side by side comparison of HNSW and LSH average search time per query metrics 

![Alt text](results_hnsw_lsh/index_mem_mb.png)
Figure 10: Side by side comparison of HNSW and LSH index memory usage


### Scaling Estimates (per query)

| method | search_est_s_per_query | notes |
| --- | --- | --- |
| HNSW | 0.000057 | log-scale factor=5.3825 efSearch=64 |
| LSH | 892.926710 | linear per-code ms=0.00089293 nbits=256 |
Table 11: Scaling estimates of LSH compared to HNSW

### Memory Estimates at Scale

| component | bytes | human |
| --- | --- | --- |
| HNSW vectors | 2048000000000 | 1.9TB |
| HNSW graph (low) | 128000000000 | 119.2GB |
| HNSW graph (high) | 256000000000 | 238.4GB |
| LSH codes | 32000000000 | 29.8GB |
| LSH total | 2080000000000 | 1.9TB |
Table 12: Scaling estimates at memory estimates.

What the estimate implies about LSH:
* LSH estimate (as implemented here = flat Hamming scan): linear in N.
approx. 892.93 s/query (approx. 14.9 min) at 1B.
This reflects FAISS IndexLSH scanning all codes—not bucketed/multi-table LSH. Proper bucketed LSH can be sublinear.

Prefer HNSW when
* Need high recall at low latency at large N.
* Want a simple recall/latency knob (efSearch) and dynamic inserts.
* Can afford the extra graph RAM (~120–240 GB at 1B) and shard across nodes.

Prefer LSH when
* Memory is extremely tight and willing to store only binary codes (no floats, no rerank) for tiny index size (≈30 GB at 1B with 256-bit codes).

* Want very fast builds and easy horizontal sharding (hash tables are naturally distributed).

* Can deploy bucketed/multi-table LSH with probing, making it sublinear in practice. Typically accept lower recall or add a small re-rank on a compact float cache.