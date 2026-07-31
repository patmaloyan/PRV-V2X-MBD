# noCheat grace/expiry study

The current configuration is **3 s grace / 6 s expiry**. The proposed grace
period was evaluated at **2 s**, first with the current 6 s expiry and then
with 4 s and 8 s expiry windows. Both noCheat attack datasets were processed
with the 1-edge and 2-edge reciprocity detectors.

## Direct comparison: current 3/6 vs proposed 2/6

| Attack | Detector | FPR | TPR | F1 |
|---|---:|---:|---:|---:|
| Random offset | 1-edge | 33.77% -> 35.23% | 94.64% -> 94.66% | 59.60% -> 58.63% |
| Random offset | 2-edge | 15.47% -> 16.36% | 94.16% -> 94.23% | 75.18% -> 74.26% |
| Constant offset | 1-edge | 36.76% -> 38.27% | 43.31% -> 45.03% | 24.65% -> 24.91% |
| Constant offset | 2-edge | 19.53% -> 20.34% | 33.76% -> 35.72% | 27.64% -> 28.49% |

The 2 s grace period detects the constant-offset attack sooner, increasing TPR
by 1.72 percentage points for 1-edge and 1.97 points for 2-edge. The cost is a
0.81-1.51 point increase in FPR. Random-offset TPR is essentially unchanged,
while F1 decreases by about 0.9 points.

## Aggregate across both attacks

| Detector | Grace / expiry | FPR | TPR | F1 |
|---|---:|---:|---:|---:|
| 1-edge | 3 / 6 s (current) | 35.33% | 73.56% | 44.39% |
| 1-edge | 2 / 4 s | 37.97% | 74.20% | 43.15% |
| 1-edge | 2 / 6 s | 36.81% | 74.28% | 43.85% |
| 1-edge | 2 / 8 s | 35.63% | 74.24% | 44.52% |
| 2-edge | 3 / 6 s (current) | 17.58% | 69.35% | 55.94% |
| 2-edge | 2 / 4 s | 17.78% | 69.20% | 55.66% |
| 2-edge | 2 / 6 s | 18.43% | 70.20% | 55.59% |
| 2-edge | 2 / 8 s | 18.98% | 70.29% | 55.10% |

## Interpretation

- For **1-edge**, 2 s grace / 8 s expiry is the best tested 2-second-grace
  compromise: aggregate F1 is slightly above current (+0.13 points), TPR rises
  0.68 points, and FPR rises only 0.30 points.
- For **2-edge**, the current 3 s / 6 s configuration still has the best
  aggregate F1 and FPR. Moving to 2 s / 6 s is justified only if the additional
  0.85 points of aggregate TPR are worth 0.85 points more FPR.
- A 2 s grace / 4 s expiry does not improve the overall 2-edge result, and it is
  clearly worse for 1-edge.

The existing current-setting result files were not overwritten. The study is
reproducible with `grace_expiry_study.py` from the visualization directory.
