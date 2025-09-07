
## Assignment 6: Analyzing and Designing your IronClad System (Part 1)

### Task 1: Compare the performance of two models and select the best performing model given the provided dataset.
Task 1 asks to compare 2 models, while restricted to only using the bruteforce indexing strategy. This analysis will explore ArcFace and VGGFace2 models using cosine distance measurement.

Baseline metrics
| Metric       | ArcFace | VGGFace2 |
| ------------ | ------: | -------: |
| Top-1 Acc    |  0.9648 |   0.9128 |
| Recall\@k    |  0.9708 |   0.9479 |
| MRR          |  0.9676 |   0.9265 |
| Precision\@k |  0.3716 |   0.3593 |
| AP\@k        |  0.9598 |   0.9096 |
| mAP\@k       |  0.9598 |   0.9096 |
| search_avg_ms|  0.0167 |   0.0096 |
Table 1: Baseline comparison of ArcFace and VGGFace2

![Alt text](storage/ArcFace_vs_VGGFace2_k5.png)
Figure 1: Side by side bar graph of ArcFace and VGGFace2 performance metrics

### Task 2: Measure the impact of Gaussian Noise adjustments on the two models' performance. Argue for which model is more robust to this type of noise, and the most performant overall.

After added embedding-space noise it seems like the arcface performance improves at sigma = 0.02 before it starts to fall at sigma =0.05 and 0.1. This is Arface demonstrates more robustnes in than vggface2 model. Arcface is also most performant overall given its higher scores across all metrics.

In tables 2 - 4 show the metric values trending downward as the value of sigma increases.

Gaussian Noise @ Sigma = 0.02
| Metric       | ArcFace | VGGFace2 |
| ------------ | ------: | -------: |
| Top-1 Acc    |  0.9658 |   0.9108 |
| Recall\@k    |  0.9708 |   0.9459 |
| MRR          |  0.9683 |   0.9248 |
| Precision\@k |  0.3716 |   0.3587 |
| AP\@k        |  0.9603 |   0.9074 |
| mAP\@k       |  0.9603 |   0.9074 |

Table 2: Model metrics with gaussian noise level of 0.2 std 


Gaussian Noise @ Sigma = 0.05
| Metric       | ArcFace | VGGFace2 |
| ------------ | ------: | -------: |
| Top-1 Acc    |  0.9627 |   0.8968 |
| Recall\@k    |  0.9688 |   0.9419 |
| MRR          |  0.9653 |   0.9145 |
| Precision\@k |  0.3706 |   0.3551 |
| AP\@k        |  0.9568 |   0.8929 |
| mAP\@k       |  0.9568 |   0.8929 |

Table 3: Model metrics with gaussian noise level of 0.05 std 


Gaussian Noise @ Sigma = 0.1
| Metric       | ArcFace | VGGFace2 |
| ------------ | ------: | -------: |
| Top-1 Acc    |  0.9386 |   0.8236 |
| Recall\@k    |  0.9567 |   0.9128 |
| MRR          |  0.9464 |   0.8601 |
| Precision\@k |  0.3654 |   0.3385 |
| AP\@k        |  0.9361 |   0.8261 |
| mAP\@k       |  0.9361 |   0.8261 |

Table 5: Model metrics with gaussian noise level of 0.1 std 


### Task 3: Measure the impact of Resizing adjustments on the two models' performance. Argue for which model is more robust to this type of noise, and the most performant overall.

As the images scale to smaller sizes, performance decreases for both models. However VGGFace2 has larder degradation in performance compared to the ArcFace model. Arcface is proving to be the more robust model against noise.

It is worth noting that these metrics are slightly higher than the baseline metrics. This could be a result of image distortion in the original dataset; and the down scaling introduced in the analysis actually improved the image quality.

Tables 6 - 8 compare the models' performance in side by side tables for each scaling value. 


Image Scale = 0.75
| Metric          | ArcFace (r100) | VGGFace2 |
| --------------- | -------------: | -------: |
| top1\_acc       |         0.9740 |   0.9268 |
| recall\@5       |         0.9790 |   0.9579 |
| precision\@5    |         0.3740 |   0.3631 |
| MRR             |         0.9765 |   0.9396 |
| AP\@5           |         0.9712 |   0.9243 |
| mAP\@5          |         0.9712 |   0.9243 |
| search\_avg\_ms |         0.0053 |   0.0050 |

Table 6: Model metrics with images scaled down to 0.75 of the original image size


Image Scale = 0.50
| Metric          | ArcFace (r100) | VGGFace2 |
| --------------- | -------------: | -------: |
| top1\_acc       |         0.9710 |   0.9186 |
| recall\@5       |         0.9760 |   0.9569 |
| precision\@5    |         0.3728 |   0.3619 |
| MRR             |         0.9735 |   0.9350 |
| AP\@5           |         0.9680 |   0.9191 |
| mAP\@5          |         0.9680 |   0.9191 |
| search\_avg\_ms |         0.0049 |   0.0054 |

Table 7: Model metrics with images scaled down to 0.50 of the original image size

Image Scale = 0.25
| Metric          | ArcFace (r100) | VGGFace2 |
| --------------- | -------------: | -------: |
| top1\_acc       |         0.9640 |   0.7876 |
| recall\@5       |         0.9740 |   0.8818 |
| precision\@5    |         0.3720 |   0.3214 |
| MRR             |         0.9682 |   0.8245 |
| AP\@5           |         0.9634 |   0.7804 |
| mAP\@5          |         0.9634 |   0.7804 |
| search\_avg\_ms |         0.0050 |   0.0048 |

Table 8: Model metrics with images scaled down to 0.25 of the original image size



### Task 4: Measure the impact of Brightness adjustments on the two models' performance. Argue for which model is more robust to this type of noise, and the most performant overall.

As brightness increase so does performance for the ArcFace model and plateus for bright levels 1.25 and 1.50. VGGFace2 increases its performance as darker images brighten up to the original brightness level, yet performance begins to decrease as the brightness increases beyond the original brightness level. Overall Arcface continues to prove it is more robust when compared to the VGGFace2

Again, it is worth noting that these metrics are a little higher than the baseline metrics. This could be an artifact of implementing the augmentation for analysis.

Tables 9 - 12 display the trends in the performance metrics as the brightness levels increase.

Image Brightness = 0.50
| Metric           | ArcFace  | VGGFace2 |
| ---------------- | -------- | -------- |
| top1\_acc        | 0.9720   | 0.9208   |
| recall\_at\_k    | 0.9780   | 0.9619   |
| precision\_at\_k | 0.3724   | 0.3635   |
| mrr              | 0.9745   | 0.9369   |
| ap\_at\_k        | 0.9682   | 0.9199   |
| map\_at\_k       | 0.9682   | 0.9199   |
| search\_avg\_ms  | 0.0048   | 0.0047   |

Table 9: Model metrics with images brightness decreased to 0.50 of the original image brightness


Image Brightness = 0.75
| Metric           | ArcFace  | VGGFace2 |
| ---------------- | -------- | -------- |
| top1\_acc        | 0.9740   | 0.9339   |
| recall\_at\_k    | 0.9790   | 0.9629   |
| precision\_at\_k | 0.3736   | 0.3645   |
| mrr              | 0.9765   | 0.9459   |
| ap\_at\_k        | 0.9702   | 0.9306   |
| map\_at\_k       | 0.9702   | 0.9306   |
| search\_avg\_ms  | 0.0054   | 0.0054   |

Table 10: Model metrics with images brightness decreased to 0.75 of the original image brightness


Image Brightness = 1.25
| Metric           | ArcFace   | VGGFace2 |
| ---------------- | --------- | -------- |
| top1\_acc        | 0.9750    | 0.9248   |
| recall\_at\_k    | 0.9810    | 0.9549   |
| precision\_at\_k | 0.3744    | 0.3625   |
| mrr              | 0.9774    | 0.9372   |
| ap\_at\_k        | 0.9718    | 0.9221   |
| map\_at\_k       | 0.9718    | 0.9221   |
| search\_avg\_ms  | 0.0055    | 0.0052   |

Table 11: Model metrics with images brightness increased to 1.25 of the original image brightness


Image Brightness = 1.50
| Metric           | ArcFace  | VGGFace2 |
| ---------------- | -------- | -------- |
| top1\_acc        | 0.9750   | 0.9147   |
| recall\_at\_k    | 0.9810   | 0.9584   |
| precision\_at\_k | 0.3744   | 0.3563   |
| mrr              | 0.9774   | 0.9322   |
| ap\_at\_k        | 0.9728   | 0.9079   |
| map\_at\_k       | 0.9728   | 0.9079   |
| search\_avg\_ms  | 0.0046   | 0.0046   |

Table 12: Model metrics with images brightness increased to 1.50 of the original image brightness



### Task 5: Define N as the final number of candidate identities the system returns to the user (i.e., Top-N nearest neighbors). Argue for the best N given the images (probe and gallery) in the dataset provided. Additionally, provide a justification for your selected value of N using performance curves or statistical summary tables (e.g., top-N accuracy vs. N).

From the available analyis of model performance on brightness there is an clear performance improvment when as the N increases. However, its clear that increase N higher than 5 will produce marginal gains at the expense of the user experience. This means more images for the security personnel to analyze and increased wait times processing. For these reasons The optimal value is 5. 

Table 13 uses the value in the brightness comparison to focus on consitant improvement across varying degrees of image augmentation. 

| Brightness | Model    | Top-1 | Top-5 (image-level) |
| ---------- | -------- | ----- | ------------------- |
| 0.50       | ArcFace  | 0.972 | 0.978               |
| 0.50       | VGGFace2 | 0.921 | 0.962               |
| 0.75       | ArcFace  | 0.974 | 0.979               |
| 0.75       | VGGFace2 | 0.934 | 0.963               |
| 1.25       | ArcFace  | 0.975 | 0.981               |
| 1.25       | VGGFace2 | 0.925 | 0.955               |
| 1.50       | ArcFace  | 0.975 | 0.981               |
| 1.50       | VGGFace2 | 0.915 | 0.958               |

Table 13: Shows the improvement in metrics with increased N value.