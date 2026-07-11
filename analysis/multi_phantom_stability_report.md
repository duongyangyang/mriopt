# Multi-Phantom Optimum Stability Check

## Setup
- Objective: `CNR_WM_GM - 1e-05 * scan_time(TR)`, scan_time = TR * 128
- 10 phantoms evaluated using the actual `simulate_mri()` function
  (not the closed-form shortcut), each with independent noise seeds per
  (phantom, TR, TE) combination — Gaussian noise std = 0.02
- Grid: TR ∈ [200,4000] step 100, TE ∈ [10,200] step 5

## Result

 phantom  TR_star  TE_star    J_max  CNR_at_opt  noise_std_est_at_opt
       0    500.0     10.0 1.089620    1.729620              0.019804
       1   3000.0     40.0 2.002326    5.842326              0.019928
       2   3000.0     30.0 5.253877    9.093877              0.019878
       3   2500.0     60.0 2.375207    5.575207              0.019925
       4    600.0     10.0 2.791617    3.559617              0.020059
       5   3500.0     50.0 2.834401    7.314401              0.019891
       6   3700.0     10.0 7.469865   12.205865              0.019884
       7    400.0     10.0 0.904541    1.416541              0.019890
       8   3700.0     10.0 2.583536    7.319536              0.019820
       9   3800.0     15.0 2.684665    7.548665              0.019871

**TR\*: mean=2470.0 ms, std=1417.4 ms, range=[400, 3800]**
**TE\*: mean=24.5 ms, std=19.2 ms, range=[10, 60]**

## Fix applied
Each phantom now gets its own (T1, T2, PD) per tissue class, sampled with
5% relative std around the base values (`get_phantom_tissue_params`),
instead of one global constant signal shared by all phantoms.

## Verdict: VARIABLE (phantom-dependent)

**With per-phantom tissue variability, TR\* varies with std=1417.4ms
(range [400, 3800]) and TE\* with
std=19.2ms (range [10, 60]).**
This variability comes directly from each phantom having its own sampled
(T1, T2, PD) per tissue class, so `mean(WM)` and `mean(GM)` genuinely
differ from phantom to phantom -- not from noise-estimation jitter
(noise_std_est stays tight: std=0.00007).
This is a physically meaningful, image-relevant target: a CNN now has real
per-phantom tissue-contrast differences to learn from.

Some phantoms land on the short-TR/short-TE local peak while others land on
the long-TR peak -- a bimodal split expected from the two-competing-peaks
landscape structure found in the earlier objective-redesign study. This is
usable learning signal, though a smoother objective could be revisited
later if CNN training proves difficult on this bimodal target.

**Conclusion: this fix is sufficient to proceed to full CNN dataset
generation.** Partial-volume effects at tissue boundaries would further
improve realism but are not required to meet the stated success criterion.

## Files produced
- `multi_phantom_grid.csv` — full (phantom, TR, TE, CNR, J) grid for all 10 phantoms
- `multi_phantom_optima.csv` — per-phantom optimum summary
- `multi_phantom_optima_scatter.png` — TR*/TE* scatter across phantoms
