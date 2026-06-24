"""
meta.py — Hypothesis metadata for Track7Harness candidate_3.

Machine-read by run_analyzer.py and loop.py. No executable code.
"""

TARGET_FAILURE_CLASSES = ["navmesh_disconnected_stair_centroid"]

TARGET_SCENES = ["q3zU7Yy5E5s", "qyAac8rV8Zk"]

HYPOTHESIS = """
The upstair centroid for q3zU7Yy5E5s at [-2.12, 3.28] is stored at detection time
without checking navigable_map connectivity. Every prior T7 interception (patch.py
candidates 1-2) attempted to fix the centroid at GCTS execution time — after the
disconnected waypoint is already committed. All T6 approaches (c2 gcts_streak=8,
c5 gcts_step=0 via floor.py __init_subclass__, c9 stair.py streak==1, c10 hooks.py
on_pre_gcts) similarly intercepted mid-GCTS with a ±50px box check — a search region
too small to span the navmesh gap at q3zU7Yy5E5s.

This candidate implements ring-expansion snap at the first GCTS call (streak==1),
sampling N_SNAP_ANGLES=16 candidates per ring at SNAP_RING_STEP=0.5m intervals up to
SNAP_MAX_DIST=3.0m (96 total candidates). This is 4× wider and 6× denser than the
prior ±50px box check, and is architecturally upstream of the native stall detector
(frontier_stick_step>=30 or gcts_step>=60).

The snap is implemented in stair.py's custom_stair_approach SDP. patch.py's
_patched_gcts wires the call at streak==1 and mutates om._up_stair_frontiers_px and
om._up_stair_frontiers so all subsequent GCTS calls use the snapped centroid.
"""

MECHANISM = """
Fix 5 in patch.py: at streak==1 (first _get_close_to_stair call), call
get_harness().custom_stair_approach(env, centroid_px, om._navigable_map,
om.pixels_per_meter). If the returned snapped pixel differs from the raw centroid,
overwrite om._up_stair_frontiers_px and om._up_stair_frontiers before _orig_gcts
receives the target. Log [T7_CENTROID_SNAP] with old and new pixel coordinates.

stair.py's snap_centroid_to_navigable: ring-expands outward from centroid_px at
SNAP_RING_STEP=0.5m, sampling N_SNAP_ANGLES=16 angles per ring up to
SNAP_MAX_DIST=3.0m. Returns np.array([col, row]) for the first navigable pixel found,
or None if the search space is exhausted.

Pixel convention: px[0]=col, px[1]=row (confirmed by obstacle_map.py:339 and T5 c24).
navigable_map indexed as navigable_map[row, col] = navigable_map[px[1], px[0]].
"""

PREDICTED_CHANGE = (
    "q3zU7Yy5E5s: snap fires at streak=1 (step ~70), centroid replaced with first "
    "navigable pixel within 3.0m. PointNav succeeds, Phase 0 completes, stair "
    "traversal proceeds. gcts_streak never reaches early-disable threshold (10). "
    "qyAac8rV8Zk: centroid already navigable (T6 baseline centroid bypass at paused=8 "
    "still works) → snap is a no-op. Other scenes unaffected."
)

PREDICTED_SR_DELTA = 0.13

WHY_ALTERNATIVES_REJECTED = """
patch.py c0/c1: streak-based early disable and BFS flood-fill in _patched_gcts both
operate at GCTS execution time and give up on the stair (disable) rather than fixing
the centroid. SR remained at 0.4.

T6 c9/c10: ±50px box check in stair.py/hooks.py wired at streak==1. The navmesh gap
at q3zU7Yy5E5s requires searching beyond 50px radius; box check found no navigable
pixel and fell through to the raw centroid. SR did not improve for this failure class.

T6 c5 (floor.py __init_subclass__): fires at gcts_step=0 before centroid is set,
but uses a navmesh presence check rather than a snap — explicitly ruled out.

dps.py DP9 carrot changes: address waypoint distance within Phase 2 (post-centroid
approach), cannot snap a disconnected Phase 1 centroid to a reachable pixel.

floor.py hysteresis fixes (c7/c8): target XB4GS9ShBRE passive detection, not the
navmesh-disconnected centroid failure class.
"""

WHY_THIS_WILL_WORK = """
q3zU7Yy5E5s centroid [-2.12, 3.28] is confirmed navmesh-disconnected. Ring expansion
at 0.5m–3.0m with 16 angles per ring covers 96 candidate pixels. Any adjacent
navigable pixel within 3.0m will allow PointNav to reach the stair entry. The snap
fires at streak==1 before the native stall detector (streak 30–60) wastes steps.

For qyAac8rV8Zk, the centroid at [-1.22, -8.19] was already handled by the
_CENTROID_BYPASS_STEPS=8 mechanism in patch.py Fix 2 (centroid bypass). The snap
either no-ops (centroid navigable) or additionally improves it without regression.

T5 c11/c36 validated that ring-expansion with px[0]=col/px[1]=row correctly handles
this coordinate convention. The 3.0m max radius and 0.5m step are conservative:
T5 c36 used RING_STEP=0.5m/MAX=3.0m/N=8; this candidate doubles N to 16 for denser
angular coverage.
"""

FALSIFIABILITY_CHECK = """
Log must contain [T7_CENTROID_SNAP] with new pixel coordinates differing from the
raw centroid for q3zU7Yy5E5s (env where q3zU7Yy5E5s scenes appear).
gcts_streak must NOT reach _N_EARLY_STAIR_DISABLE=10 for that scene.
Phase 0 (_reach_stair) must be entered and _climb_stair must be called for q3zU7Yy5E5s.
"""
