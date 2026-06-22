| stage | description | equation |
| --- | --- | --- |
| Stage 0 | Sensors-style reproduction Stage 3 with CTL readiness and progress fatigue | t = d 3600 f_GAP / (v_VT2 alpha f_alt f_CTL f_pad(p)) |
| Stage 1 | Replace progress fatigue with decayed acute TRIMP fatigue | t = d 3600 f_GAP / (VMA alpha f_alt f_CTL F(D)) |
| Stage 2 | Replace CTL readiness with REDI readiness | t = d 3600 f_GAP / (VMA alpha f_alt f_REDI F(D)) |
| Stage 3 | Add fixed linear HRR speed ratio and select decayed or cumulative acute TRIMP | t = d 3600 f_GAP / (VMA alpha f_alt f_REDI E(HRR) F(U)), U in {D, C} |
