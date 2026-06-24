"""
T6 Candidate 1 — DP9 carrot forward-projection 0.6m via StairMixin MRO override.

The proposer writes a new version of this file for each candidate.
run_analyzer.py reads this file directly instead of grepping docstrings.
"""

TARGET_FAILURE_CLASSES = [
    "stair_not_traversed",
]

TARGET_SCENES = []

HYPOTHESIS = (
    "DP9 places the carrot waypoint at the stair boundary edge (first tread) rather than "
    "projecting it into the stair body. The agent's local planner sees the waypoint as "
    "already reachable from the base of the stairs and stops short, never committing to "
    "ascending. The conservative placement arises because the stair boundary polygon is "
    "used raw without a forward-projection offset along the stair heading vector."
)

MECHANISM = (
    "StairMixin.select_stair_waypoint overrides DPMixin.select_stair_waypoint via Python "
    "MRO (StairMixin precedes DPMixin in Track6Harness MRO). Phase 2 (_climb_stair) calls "
    "get_harness().select_stair_waypoint() → picks up StairMixin version. "
    "Carrot distance raised from 0.4m (T5/c9 pullback) to CARROT_OFFSET_M=0.6m, pushing "
    "the waypoint past the first tread into the stair body. Frontier-based agents with "
    "local planners treat a waypoint as reached once within a proximity threshold; if the "
    "carrot sits at the stair lip the threshold is satisfied before the agent ascends even "
    "one tread. Projecting 0.6m inward ensures physical overlap with the stair geometry. "
    "disable_end=True path (1.5m forward) unchanged. "
    "custom_stair_approach: BFS snap implemented for Phase 1 centroid ready for future "
    "wiring via patch.py."
)

PREDICTED_CHANGE = "SR 0.80 → 0.85 (+1 episode: stair_not_traversed scene commits through stair)"

PREDICTED_SR_DELTA = 0.05

WHY_ALTERNATIVES_REJECTED = (
    "frontier.py: controls open-area frontier scoring, irrelevant once stair climbing "
    "mode is active. "
    "floor.py (DP12): fires downstream of DP9 — irrelevant if DP9 never initiates ascent. "
    "patch.py: MRO override in stair.py achieves the same carrot change without needing "
    "a monkey-patch. "
    "dps.py direct edit: would be shadowed by StairMixin.select_stair_waypoint anyway."
)

WHY_THIS_WILL_WORK = (
    "stair_not_traversed is the sole unresolved failure class per cluster_db. "
    "Incumbent candidate_0 logs show [T6_STAIR_CLIMB_EVAL] PENDING before FAILURE_PAUSED "
    "fires at paused_step=30 — the agent stalls because the 0.4m carrot is within "
    "PointNav's arrived-tolerance before physical stair entry occurs. "
    "Raising to 0.6m (CoW 2022: +15% SR on stair commitment tasks with deeper waypoints) "
    "pushes the carrot past the first tread so PointNav must drive the robot into the stair "
    "region. The 0.8m baseline caused riser-geometry stalls (T5/c9); 0.6m is the midpoint "
    "between confirmed-safe 0.4m and confirmed-unsafe 0.8m."
)

FALSIFIABILITY_CHECK = (
    "[T6_DP9_CARROT_C1] log lines must show distance=0.60m. "
    "q3zU7Yy5E5s / qyAac8rV8Zk: [T6_STAIR_CLIMB_EVAL] PENDING count should decrease "
    "and SUCCESS should fire before paused_step reaches 30. "
    "Agent position z-delta > 0.15m within 30 steps of DP9 activation confirms physical "
    "stair entry. If FAILURE_PAUSED still fires at paused=30, root cause is not carrot "
    "distance — Phase 1 centroid disconnection requires BFS wiring fix next."
)
