| # | Target | Start (x, y, yaw) | Start dist (m) | Condition | Result | t (s) | d (m) |
|---|---|---|---|---|---|---|---|
| 1 | green chair | (0.0, 0.0, 0) | 3.23 | visible; red chair also in view | SUCCESS | 6.4 | 0.55 |
| 2 | red chair | (0.0, 0.0, 0) | 3.23 | visible; green chair also in view | SUCCESS | 6.4 | 0.59 |
| 3 | blue chair | (0.0, 0.0, 0) | 2.92 | not initially visible (behind) | SUCCESS | 9.2 | 0.51 |
| 4 | stop sign | (0.0, 0.0, 0) | 3.35 | not initially visible; no colour given | SUCCESS | 12.4 | 0.58 |
| 5 | green chair | (0.0, 0.0, 180) | 3.23 | facing away | SUCCESS | 10.9 | 0.52 |
| 6 | red chair | (1.0, -2.5, 90) | 2.39 | side start | SUCCESS | 12.6 | 0.6 |
| 7 | blue chair | (0.5, 2.5, -90) | 3.16 | not initially visible | SUCCESS | 13.0 | 0.6 |
| 8 | red stop sign | (1.0, 0.0, 90) | 3.91 | side start | SUCCESS | 10.9 | 0.5 |
| 9 | green chair | (-1.0, -1.0, 45) | 4.57 | diagonal start | SUCCESS | 16.7 | 0.6 |
| 10 | red chair | (1.5, 0.3, -20) | 2.12 | close start, green chair nearby | SUCCESS | 4.8 | 0.58 |
| 11 | blue chair | (-4.5, -0.5, 0) | 2.83 | long side start | SUCCESS | 14.6 | 0.79 |
| 12 | yellow chair | (0.0, 0.0, 0) | - | target absent (negative test) | FAIL (not_found_after_full_turn) | 10.0 | - |
| 13 | red chair | (-1.7, 1.0, 148) | 5.19 | far: after blue-chair mission | SUCCESS | 29.7 | 0.55 |
| 14 | blue chair | (2.4, -0.8, -34) | 5.41 | far: after red-chair mission | FAIL (timeout) | 60.1 | 5.38 |
| 15 | blue chair | (2.6, 0.9, 37) | 5.14 | far: after green-chair mission | SUCCESS | 13.7 | 0.57 |
| 16 | green chair | (-1.2, -2.5, -121) | 5.6 | far: after stop-sign mission | SUCCESS | 18.8 | 0.59 |

| Metric | Near starts (2.1-4.6 m) | Far starts (5.1-5.6 m) | All positive trials |
|---|---|---|---|
| Detection (target seen) | 11/11 (100%) | 4/4 (100%) | 15/15 (100%) |
| Grounding (right object chosen) | 11/11 (100%) | 3/4 (75%) | 14/15 (93%) |
| Approach success (SUCCESS and d <= 0.80 m) | 11/11 (100%) | 3/4 (75%) | 14/15 (93%) |
| SUCCESS reported but d > 0.80 m | 0/11 (0%) | 0/4 (0%) | 0/15 (0%) |
| Final d of successes, mean (min-max) | 0.58 m (0.50-0.79) | 0.57 m (0.55-0.59) | 0.58 m (0.50-0.79) |
| Time to found, mean (min-max) | 10.7 s (4.8-16.7) | 20.7 s (13.7-29.7) | 12.9 s (4.8-29.7) |

Correct FAIL for the absent target: 1/1 (100%)

Failed positive trials:
- Trial 14 (blue chair, start 5.41 m away): FAIL timeout, d=5.38 m
