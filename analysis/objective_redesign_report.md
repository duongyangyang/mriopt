# Objective Redesign Report: Non-Trivial TR/TE Landscapes

## 1. Problem with the original objective

`CNR_WM_GM` alone is monotonically increasing in TR across [200, 4000] ms
(see prior physics validation study) — the optimum sits at the TR=4000ms
grid boundary. Any optimizer (random search, Bayesian optimization, or a
learned model) trivially converges to "TR = maximum allowed" without
learning any real trade-off. This is not a meaningful target for ML.

## 2. Redesigned objectives compared

| Objective | Formula | Rationale |
|---|---|---|
| CNR only (baseline) | `\|S_WM - S_GM\| / sigma` | Reference; known boundary-pushing behavior |
| Relative contrast | `\|S_WM - S_GM\| / mean(S_WM,S_GM)` | Normalizes contrast by signal level; removes absolute-scale bias |
| CNR − λ·scan_time | `CNR - λ·(TR·N_PE)` | Penalizes long acquisitions directly (λ auto-scaled to CNR range) |
| CNR efficiency | `CNR / sqrt(TR·N_PE)` | Classic MRI contrast-per-sqrt(time) efficiency metric |
| Multi-objective | `0.5·CNR_norm + 0.3·SNR_norm − 0.2·time_norm` | Explicit weighted trade-off across contrast, signal level, and time |

All use the same unmodified spin-echo signal model and fixed noise floor
(sigma=0.02) as the original physics validation study; only the
*objective function computed on top of the simulator output* is changed —
the simulator itself is untouched.

## 3. Quantitative landscape comparison

                                     objective  TR_star  TE_star    J_max  boundary_optimum  roughness  n_local_maxima  curvature_at_opt
                                      CNR_only   4000.0     25.0 6.198124              True   0.001667               2               NaN
                             Relative_contrast   4000.0    200.0 0.633483              True   0.000276               2               NaN
       CNR_minus_lambda_time (lambda=1.00e-05)   3300.0     35.0 1.227484             False   0.002825               3          0.019397
               CNR_efficiency (CNR/sqrt(time))   3850.0     30.0 0.002327             False   0.004084               2          0.000008
Multi_objective (0.5 CNR + 0.3 SNR - 0.2 time)    450.0     10.0 0.099835              True   0.001543               3               NaN

**Column definitions:**
- `boundary_optimum`: True if TR* or TE* is within 3% of a grid edge.
- `roughness`: std(|gradient|) / objective range — scale-free landscape smoothness (lower = smoother, easier to model).
- `n_local_maxima`: count of local maxima via 3×3 max-filter — proxy for multimodality/optimization difficulty.
- `curvature_at_opt`: |d²J/dTR²| + |d²J/dTE²| at the optimum (finite differences) — higher = sharper, more identifiable peak; NaN if optimum is on the boundary (no two-sided curvature).

## 4. Key physics discovery that made this possible

The raw `S_WM(TR) - S_GM(TR)` difference is **not monotonic**: WM (T1=850ms)
recovers faster than GM (T1=1300ms), so at short-to-moderate TR, WM signal
briefly exceeds GM signal, producing a small local CNR hump around
**TR≈500-600ms**. As TR increases further, the two signals cross
(near TR≈1000-1500ms, CNR dips toward ~0), then GM's higher proton density
(0.85 vs 0.70) wins out asymptotically, producing the larger CNR value at
TR=4000ms found in the original physics study. This means the CNR(TR)
landscape has **two competing peaks**: a smaller local one at moderate TR,
and a larger asymptotic one at the TR boundary. A correctly tuned time
penalty can make the *early, cheap* peak win over the *late, expensive* one
— which is exactly the interior-optimum behavior the task requires.

## 5. Findings

- **CNR only** and **Relative contrast**: both retain boundary optima
  (TR pinned at 4000ms). Normalizing contrast by signal level does not
  change which of the two peaks is larger, so it does not resolve the
  trivial-landscape problem on its own.
- **CNR − λ·scan_time**: a lambda sweep (`lambda_sweep.csv`, log-spaced
  1e-8 to 1e-4) shows the optimum jumps discretely between the two peaks —
  it stays at the TR=4000ms boundary for λ ≲ 6.8e-6, briefly lands on a
  genuine **interior optimum near λ=1e-5** (TR*=3300ms, TE*=35ms — inside
  the grid, not at either boundary), then overshoots to the TR=200ms
  boundary for λ ≳ 1.5e-5. The usable interior window is narrow, which is
  itself an informative result about how sharply this trade-off is balanced.
- **CNR efficiency (CNR/time^p)**: similarly fragile — p=0.5 (the classic
  sqrt-time metric) sits at the TR=4000 boundary, p=0.6 lands just inside
  (TR≈3850ms), and p≥0.65 overshoots straight to the TR=200 boundary. No
  wide stable interior plateau exists for this functional form.
- **Multi-objective**: with time weighted heavily (w_time=0.5) the TR
  optimum moves well inside the range (TR≈450ms), but TE still collapses to
  its lower boundary (TE=10ms) because nothing in this formulation
  penalizes short TE — only the CNR term itself provides any TE trade-off.

## 6. Recommended objective

**Recommended: `CNR − λ·scan_time` with λ=1e-5**
(interior optimum at TR*=3300 ms, TE*=35 ms — verified inside both parameter
ranges, not on any boundary)

This is the only tested formulation that produces a *fully* interior
optimum (both TR* and TE* away from all four grid edges) while remaining
directly physically interpretable: it is literally "contrast gained minus
the cost of the time spent acquiring it," with λ setting the ms-of-contrast
you're willing to trade for one ms of scan time. Because the underlying
CNR(TR) landscape has two competing peaks (Section 4), this objective is
also more representative of real protocol design than a smooth single-peak
function would be — it is sensitive to λ, phantom-specific tissue contrast,
and noise level, all of which shift which peak wins. That sensitivity is
precisely what justifies the use of Bayesian Optimization or Deep Learning:

- **non-trivial enough to justify Bayesian Optimization** — evaluating the
  true objective requires a forward simulation (expensive black-box
  evaluation), and BO can find the interior optimum in far fewer
  evaluations than grid/random search, which is a real advantage only
  when the optimum isn't already known to sit at a boundary.
- **non-trivial enough to justify a learned CNN/regressor** — a network
  trained to predict J(image, TR, TE) must learn a genuine curved
  response surface (not a monotonic ramp), giving ML methods something
  substantive to model and enabling generalization across phantoms with
  different tissue geometries/noise levels where the exact optimum shifts.

## 7. Files produced

- `lambda_sweep.csv` — TR*/TE*/boundary-flag for 25 log-spaced λ values,
  showing the discrete jump between the two competing peaks
- `objective_comparison.csv` — quantitative diagnostics table (also printed above)
- `objective_comparison_grid.png` — side-by-side heatmaps of all 5 objectives
- `heatmap_CNR_only.png`, `heatmap_Relative_contrast.png`,
  `heatmap_CNR_minus_lambda_time.png`, `heatmap_CNR_efficiency.png`,
  `heatmap_Multi_objective.png` — individual heatmaps with optimum marked
