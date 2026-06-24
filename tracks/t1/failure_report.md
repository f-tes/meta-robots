# ASCENT Failure Classification Report

_Generated: 2026-05-18 07:52 UTC_

## Candidate Summary

| Candidate | SR | #Episodes | #Failed | Primary Failure Class |
|-----------|-----|-----------|---------|----------------------|
| candidate_3 | 0.00 | 8 | 8 | search_oscillation |
| candidate_4 | 0.00 | 8 | 8 | search_oscillation |
| candidate_5 | 0.00 | 8 | 8 | search_oscillation |
| candidate_6 | 0.50 | 8 | 4 | navigation_stair_traverse |
| candidate_7 | 0.50 | 8 | 4 | navigation_stair_traverse |
| candidate_8 | 0.00 | 1 | 1 | search_oscillation |
| candidate_9 | 0.00 | 1 | 1 | search_oscillation |

## Global Failure Class Counts (all candidates combined)

| Rank | Failure Class | Count | % of All Failures |
|------|---------------|-------|------------------|
| 1 | `search_oscillation` | 24 | 70.6% |
| 2 | `navigation_stair_traverse` | 10 | 29.4% |

## Top 3 Failure Classes — Detail

### `search_oscillation` (24 episodes)

- **Typical step count:** 138
- **Typical floor re-inits:** 1.7
- **Example episodes:**
  - candidate_3 / DYehNKdT76V (chair)
  - candidate_3 / p53SfW6mjZe (tv)
  - candidate_3 / wcojb4TFT35 (chair)

### `navigation_stair_traverse` (10 episodes)

- **Typical step count:** 330
- **Typical floor re-inits:** 1.5
- **Example episodes:**
  - candidate_3 / q3zU7Yy5E5s (couch)
  - candidate_3 / qyAac8rV8Zk (couch)
  - candidate_4 / q3zU7Yy5E5s (couch)

## Recommendation for Next Candidate

**Most frequent unresolved failure class:** `search_oscillation`
(24 of 34 failed episodes, 71%)

No specific recommendation available for this failure class.
