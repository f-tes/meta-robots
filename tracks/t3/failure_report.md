# ASCENT Failure Classification Report

_Generated: 2026-05-22 07:27 UTC_

## Candidate Summary

| Candidate | SR | #Episodes | #Failed | Primary Failure Class |
|-----------|-----|-----------|---------|----------------------|
| candidate_0 | 0.50 | 10 | 5 | mapping_floor_confusion |

## Global Failure Class Counts (all candidates combined)

| Rank | Failure Class | Count | % of All Failures |
|------|---------------|-------|------------------|
| 1 | `mapping_floor_confusion` | 3 | 60.0% |
| 2 | `navigation_stair_traverse` | 2 | 40.0% |

## Top 3 Failure Classes — Detail

### `mapping_floor_confusion` (3 episodes)

- **Typical step count:** 291
- **Typical floor re-inits:** 3.0
- **Example episodes:**
  - candidate_0 / 4ok3usBNeis (tv)
  - candidate_0 / mL8ThkuaVTM (toilet)
  - candidate_0 / XB4GS9ShBRE (bed)

### `navigation_stair_traverse` (2 episodes)

- **Typical step count:** 328
- **Typical floor re-inits:** 1.5
- **Example episodes:**
  - candidate_0 / qyAac8rV8Zk (couch)
  - candidate_0 / q3zU7Yy5E5s (couch)

## Recommendation for Next Candidate

**Most frequent unresolved failure class:** `mapping_floor_confusion`
(3 of 5 failed episodes, 60%)

ASCENT's elevation-based floor detector is triggering spurious re-initialisations.  The agent never settles on a floor long enough to map it.  Patch the floor-change hysteresis in `ascent_policy.py` (Track 2) or increase DP12 minimum interval.
