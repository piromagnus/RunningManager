**Table 7. Slight near-flat immobile segment rejection policies and LOO impact.**

| Policy | Thresholds | Segments total | Segments rejected | Rejected share (%) | Rejected time (min) | Mean Stage-3 LOO MAE (min) |
| --- | --- | --- | --- | --- | --- | --- |
| None (baseline) | exclusion disabled | 2364 | 0 | 0.0 | 0.0 | 10.87 |
| Slight (paper default) | speedEq < 3 km·h⁻¹ or stationary share > 0.40; |Δelev|/h ≤ 120 m·h⁻¹ | 2364 | 48 | 2.03 | 1102.4 | 11.85 |
| Moderate | speedEq < 4 km·h⁻¹ or stationary share > 0.30; |Δelev|/h ≤ 120 m·h⁻¹ | 2364 | 101 | 4.27 | 1919.3 | 10.3 |
| Slight + moving-time fit (paper §7) | speedEq < 3 km·h⁻¹ or share > 0.40; fit on moving time | 2364 | 4 | 0.17 | 84.1 | 9.09 |

*Rejection requires altitude–time flatness (|Δelev|/h ≤ 120 m·h⁻¹) and immobility (low grade-adjusted speed or high stationary share). Rejected segments are withheld from parameter fitting but retained for full-race evaluation. The paper §7 pipeline combines slight exclusion with moving-time fitting, leaving only a few residual rejects.*
