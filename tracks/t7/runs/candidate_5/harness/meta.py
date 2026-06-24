"""
meta.py — Machine-readable hypothesis metadata for candidate_5.

Read by run_analyzer.py and propose.py. No executable code.
"""

TARGET_FAILURE_CLASSES = ["navmesh_disconnected_stair_centroid"]

TARGET_SCENES = ["q3zU7Yy5E5s", "qyAac8rV8Zk"]

HYPOTHESIS = """
The upstair centroid for q3zU7Yy5E5s at [-2.12027027, 3.27567568] lies in a
disconnected navmesh component. All prior candidates (T4 c36, T5 c10/c11/c19/
c21/c24/c36, T6 c2/c5/c9/c10, T7 c1-c4) address this at Phase-1 time (gcts
streak=1 or streak=10). candidate_3 (Fix 5) snaps at streak==1 with a 3.0m ring
and candidate_3 (Fix 4) disables at streak==10 — both fire AFTER Phase 1 has
started, wasting 1-10 steps per attempt and allowing the LLM planner to select
the stair centroid as a navigation goal based on its raw (non-navigable) world
coordinates.

candidate_5 intercepts at CENTROID REGISTRATION TIME: ObstacleMap.update_map
transitions _has_up_stair from False → True. A module-level hook in hooks.py
fires _run_centroid_precheck() at this exact moment — before the LLM planner
sees the centroid, before the policy selects look_for_upstair mode, and before
any gcts step is taken. The precheck checks navigability of the centroid pixel
and ring-expands outward up to 1.4m. If no navigable pixel is found, the stair
is DISABLED immediately via _disabled_stair_map masking (persistent across
subsequent update_map calls) — Phase 1 never starts. If a navigable pixel is
found, the centroid is snapped in-place (the LLM sees the corrected centroid
from first detection).
"""

MECHANISM = """
hooks.py adds two module-level functions:

1. _install_precheck_hook(): Patches ObstacleMap.update_map to detect the
   _has_up_stair False→True transition, and ObstacleMap.reset to clear the
   per-instance _t7_precheck_done flag at episode start. Guard:
   ObstacleMap._t7_precheck_installed prevents double-wrapping on harness reload.

2. _run_centroid_precheck(om): Checks om._navigable_map[row, col] at centroid
   pixel (px[0]=col, px[1]=row per obstacle_map.py:339 convention). If
   navigable: logs [T7_CENTROID_PRECHECK] PASS, no action. If non-navigable:
   ring-expands at PRECHECK_RING_STEP_M=0.2m steps up to PRECHECK_RADIUS_M=1.4m
   (7 rings × 12 angles = 84 candidates). If a navigable pixel is found: snaps
   om._up_stair_frontiers_px and om._up_stair_frontiers in-place (LLM sees
   corrected centroid). If none found: disables stair by marking current
   om._up_stair_map pixels in om._disabled_stair_map (persistent masking via
   obstacle_map.py:576), zeroes om._up_stair_map and om._up_stair_frontiers_px,
   sets om._has_up_stair=False, adds frontier to om._disabled_frontiers.
   Log tag: [T7_CENTROID_PRECHECK].

HooksMixin.on_episode_start: preserved baseline behavior (counter increment +
telemetry write). All other HooksMixin methods: unchanged baseline.

Files changed vs candidate_3: hooks.py only.
"""

PREDICTED_CHANGE = (
    "q3zU7Yy5E5s: [T7_CENTROID_PRECHECK] DISABLED fires before any gcts step. "
    "_has_up_stair=False after precheck. Phase 1 never starts. Agent explores "
    "downstairs path. Saves 10-76 wasted Phase-1 steps vs candidate_3. "
    "qyAac8rV8Zk: centroid likely navigable → precheck PASS, Fix 5 handles as before. "
    "XB4GS9ShBRE: not targeted, candidate_3 Fix 1/2/3 unchanged."
)

PREDICTED_SR_DELTA = 0.133

WHY_THIS_WILL_WORK = """
For q3zU7Yy5E5s: the centroid disconnection is confirmed by 6+ prior tracks.
All prior fixes reacted at Phase-1 time (streak=1 or streak=10). The precheck
fires at centroid REGISTRATION time — potentially 10-50 steps before the LLM
directs the agent to the stair. The _disabled_stair_map persistent masking
(obstacle_map.py:576) ensures the stair won't be re-detected in the same pixel
region after disable. No Phase-1 entry = zero wasted steps per stair attempt.

candidate_3 (SR=0.4333) improved over candidate_0 (SR=0.4) via Fix 4+5. The
precheck fires even earlier, removing the residual gcts overhead.

For qyAac8rV8Zk: centroid already navigable → precheck PASS → no change vs c3.
"""

WHY_ALTERNATIVES_REJECTED = """
patch.py: all prior T7 patch.py changes at gcts streak=1 or streak=10 fire
AFTER Phase 1 starts. Moving the check to centroid registration time requires
a different hook point. Candidate_3 (incumbent) already has the best patch.py
approach (Fix 4+5).

floor.py: candidate_4's check_upstair_centroid_navigable in floor.py also fires
at streak==1 (via patch.py Fix 6 call). candidate_4 SR=0.3667 — regression vs
candidate_3. The strict single-pixel navcheck was too aggressive (blocked
borderline-navigable centroids for other scenes). Our hooks.py approach avoids
modifying patch.py (uses ring-expansion not strict-pixel) and fires earlier.

stair.py: custom_stair_approach (candidate_3 Fix 5) already fires at streak==1.
A module-level hook in hooks.py fires earlier (at update_map), without needing
to modify stair.py or patch.py.
"""

FALSIFIABILITY_CHECK = """
Log MUST contain '[T7_CENTROID_PRECHECK] ... DISABLED' for q3zU7Yy5E5s episodes.
Log MUST NOT contain 'look_for_upstair' mode entries after the [T7_CENTROID_PRECHECK]
DISABLED line for q3zU7Yy5E5s.
gcts_streak for q3zU7Yy5E5s MUST be 0 (Phase 1 never entered).
SR must exceed 0.4333 (candidate_3 incumbent).
"""
