| stage | description | equation |
| --- | --- | --- |
| Stage 0 | Sensors-style reproduction Stage 3 with CTL readiness and progress fatigue | t = d 3600 f_GAP / (v_VT2 alpha f_alt f_CTL f_pad(p)) |
| Stage 1 | Replace progress fatigue with raw decayed acute TRIMP fatigue | t = d 3600 f_GAP / (VMA alpha f_alt f_CTL F(D_raw)) |
| Stage 2 | Replace CTL readiness with REDI readiness | t = d 3600 f_GAP / (VMA alpha f_alt f_REDI F(D)) |
| Stage 3 | Add fixed linear HRR speed ratio and select raw decayed TRIMP, raw cumulative TRIMP, progress fatigue, or decayed-plus-muscular fatigue | t = d 3600 f_GAP / (VMA alpha f_alt f_REDI E(HRR) F(U)), U in {D_raw, C_raw, progress}; combined uses F(D_raw,M) |
| Stage 3 fatigue | Linear/exponential fatigue use the selected raw load directly; combined fatigue adds a fitted muscular coefficient gamma; no acute fatigue is kappa=0 | F_linear(U)=clip(1-kappa U), F_exp(U)=clip(exp(-kappa U)), F_combined(D,M)=clip(exp(-kappa D) exp(-gamma M)), F_none(U)=1 |
