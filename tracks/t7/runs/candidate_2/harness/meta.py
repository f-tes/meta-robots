"""
meta.py — machine-readable hypothesis metadata for Track7Harness candidate_2.

Read by run_analyzer.py and classify_failures.py. No executable code here.
"""

TARGET_FAILURE_CLASSES = ["navmesh_disconnected_stair_centroid"]

TARGET_SCENES = ["q3zU7Yy5E5s", "qyAac8rV8Zk"]

HYPOTHESIS = """
In _climb_stair Phase 1 (ascent_policy.py:1108-1112), any PointNav STOP is treated as
'centroid reached' and forces Phase 2. But for q3zU7Yy5E5s the upstair centroid at world
[-2.12, 3.28] is in a navmesh-disconnected component: PointNav returns STOP immediately
because it cannot plan a path, while rho (distance to centroid) remains >> 0.3m. The
existing code cannot distinguish 'STOP because near centroid (rho<=0.3m)' from 'STOP
because centroid is unreachable (rho>>0.3m)'. All prior fixes tried to redirect or bypass
after GCTS was already stuck; this is the first fix that reads the PointNav STOP signal
itself as a reachability oracle at Phase 1 entry.
"""

MECHANISM = """
Fix 5 in patch.py: monkey-patch _climb_stair via _patched_climb_stair. Before calling
_orig_climb_stair in Phase 1 (not centroid reached, bypass not fired), pre-compute
pre_rho = ||stair_centroid - robot_xy||. After _orig_climb_stair returns, check if
_reach_stair_centroid flipped True AND pre_rho > UNREACHABLE_MIN_RHO=0.8m: if so,
log [T7_GCTS_PHASE1_UNREACHABLE], call mc._disable_stair_and_reset_state(env, centroid),
return _explore(). The existing centroid bypass (Fix 2, paused>=8) is preserved inside
_patched_climb_stair. Incumbent Fix 4 (gcts_streak early disable at step 10) is unchanged.
"""

PREDICTED_CHANGE = """
q3zU7Yy5E5s: [T7_GCTS_PHASE1_UNREACHABLE] fires within 1-2 steps of entering _climb_stair
Phase 1. Agent disables upstair and transitions to downstair path (confirmed reachable in T5
c8: paused_step=22-24). XB4GS9ShBRE and qyAac8rV8Zk: pre_rho <= 0.8m or PointNav does not
return STOP at Phase 1 entry → Fix 5 never fires → behaviour unchanged from candidate_0.
"""

PREDICTED_SR_DELTA = 0.07

WHY_THIS_WILL_WORK = """
For q3zU7Yy5E5s: PointNav returns STOP on step 1 of climb_stair Phase 1 because the upstair
centroid [-2.12, 3.28] is in a disconnected navmesh island — rho at Phase 1 entry is ~2-4m.
The UNREACHABLE_MIN_RHO=0.8m gate cleanly separates this from the legitimate centroid-reached
case (rho<=0.3m per map_controller.py:279). Disabling immediately prevents Phase 2 carrot
strategy from executing on an unreachable island. For qyAac8rV8Zk: candidate_0 solved it via
centroid bypass + 0.4m carrot (dps.py baseline). pre_rho at climb_stair entry for qyAac8rV8Zk
should be <=0.8m when PointNav fires STOP (or STOP not fired), so Fix 5 stays dormant.
The rho value is computed with a single np.linalg.norm call — no PointNav invocation, no
overhead on the non-triggering path.
"""

WHY_ALTERNATIVES_REJECTED = """
gcts_streak_based_early_disable (all prior T6 candidates including c3): counts elapsed
_get_close_to_stair steps, not the PointNav planner's own reachability signal. Fires N steps
after stall begins rather than at first STOP signal. floor_py_init_subclass_injection with
50px_box_navmesh_check (T6 c5): used _navigable_map (obstacle-grid estimate) not Habitat
navmesh; 50px box too permissive, hook fired too late. Phase2_BFS_snap (T6 c2): intervenes
after Phase 1 completes. check_navigability_at_stair_centroid_registration_time using
_navigable_map is unreliable because the centroid pixel may appear free even if Habitat navmesh
is disconnected. PointNav STOP at large rho is the ground-truth reachability signal from the
actual planner — no other proxy has this fidelity.
"""

FALSIFIABILITY_CHECK = """
q3zU7Yy5E5s episode logs must contain '[T7_GCTS_PHASE1_UNREACHABLE]' within the first 3 steps
after _climb_stair is entered (pre_rho logged must be > 0.8m). The agent must NOT subsequently
re-enter look_for_upstair or get_close_to_stair for that stair instance.
qyAac8rV8Zk must retain candidate_0's SUCCESS path: no [T7_GCTS_PHASE1_UNREACHABLE] log,
downstair centroid reachable so rho<=0.8m when STOP fires (or STOP never fires).
XB4GS9ShBRE must show no regression: Fix 5 never fires because centroid is reachable.
"""
