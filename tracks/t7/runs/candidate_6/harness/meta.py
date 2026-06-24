"""
meta.py — Candidate metadata for Track7Harness candidate_6.
"""

TARGET_FAILURE_CLASSES = ["navmesh_disconnected_stair_centroid"]
TARGET_SCENES = ["q3zU7Yy5E5s", "XB4GS9ShBRE", "bxsVRursffK", "p53SfW6mjZe"]

HYPOTHESIS = """
The upstair centroid for q3zU7Yy5E5s ([-2.12, 3.28]) and other scenes lies in a
navmesh-disconnected component. Candidate_3 Fix 5 snaps at streak==1 (first GCTS
call, 3.0m ring), and Fix 4 aborts at streak==10, but these fire AFTER the centroid
has been committed for at least one PointNav attempt. Candidate_5 tried
registration-time patching in hooks.py at 1.4m ring radius — same SR (0.4333).

The untested structural point: registration-time snap in stair.py with the SAME
3.0m radius as Fix 5, wired via a module-level patch to ObstacleMap.update_map
at import time. This fires at the FIRST update_map call when _has_up_stair
transitions False→True — strictly before streak==1 and before Phase 1 is entered.

Key difference from candidate_5 (hooks.py, 1.4m): uses snap_centroid_to_navigable
(3.0m radius, 16 angles, 0.5m step) instead of 1.4m, 12 angles, 0.2m step.
A navigable pixel between 1.4m and 3.0m that candidate_5 missed may be reachable.
Guard name _t7_crsnap_installed / _t7_crsnap_done avoids conflict with anything.
"""

MECHANISM = """
stair.py adds two module-level functions at the bottom of the file (before
class StairMixin):

1. _apply_centroid_reg_snap(om_self): checks om._navigable_map[row, col] for the
   registered centroid (px[0]=col, px[1]=row convention). If navigable: PASS.
   If non-navigable: calls snap_centroid_to_navigable (already in stair.py) with
   _SNAP_MAX_DIST=3.0m, _N_SNAP_ANGLES=16, _SNAP_RING_STEP=0.5m to find nearest
   navigable pixel. If found: mutates om._up_stair_frontiers_px and
   om._up_stair_frontiers in-place. If none found within 3.0m: adds stair pixels
   to _disabled_stair_map (persistent masking in update_map:576), zeros
   _up_stair_map and frontiers, sets _has_up_stair=False. Log: [T7_CENTROID_REG_SNAP].

2. _install_centroid_reg_snap(): patches ObstacleMap.update_map and .reset at
   import time. Guard: ObstacleMap._t7_crsnap_installed prevents double-wrapping.
   Per-instance _t7_crsnap_done cleared by wrapped reset. Fires when
   _has_up_stair transitions False→True AND _t7_crsnap_done is False.

_install_centroid_reg_snap() called at module level (stair.py bottom).

patch.py Fix 5 (streak==1 snap) remains as a safety net in case navigable_map
is incomplete at first detection but complete by GCTS time.
"""

PREDICTED_CHANGE = "SR 0.4333 → 0.5000 (+0.0667): fix 2 of 4 XB4GS9ShBRE navmesh_disconnection episodes"

PREDICTED_SR_DELTA = 0.067

WHY_ALTERNATIVES_REJECTED = """
patch.py (c1, c2): gcts-streak intercepts fire after 1+ wasted PointNav steps;
  centroid already committed to Phase 1 before any check.
floor.py (c4): SR regressed to 0.3667; floor-switch hysteresis unrelated to
  pre-Phase-1 stair centroid storage.
hooks.py (c5): same registration-time approach but 1.4m radius only; tied c3 at
  0.4333. XB4GS9ShBRE navmesh_disconnection failures persist (31-76 consecutive
  Reach_stair_centroid: False) — centroid either passed 1.4m check (navigable)
  or snapped to a secondary disconnected island within 1.4m.
patch.py Fix 5 streak==1 (c3): fires at first GCTS call — one step after
  registration. Achieves 0.4333. Registration-time (c6) fires strictly earlier.
dps.py / DP9: modifies waypoint carrot distance, not centroid coordinates.
"""

WHY_THIS_WILL_WORK = """
XB4GS9ShBRE navmesh_disconnection failures persist in candidate_5 (47/52/76/31
consecutive Reach_stair_centroid: False). candidate_5's 1.4m ring either:
  (a) found a navigable pixel that is itself in a disconnected island → still stalls, or
  (b) passed (centroid navigable per pixel lookup but still disconnected by navmesh
      geometry) → stalls without snap.
Increasing radius to 3.0m expands the search to pixels that are genuinely connected
to the main navigable component, giving PointNav a reachable target. The same
snap_centroid_to_navigable function already improved SR from 0.40→0.4333 in c3
when wired at streak==1; wiring it earlier (registration-time) and using the same
3.0m radius avoids 1+ wasted PointNav-to-disconnected-centroid attempts.
"""

FALSIFIABILITY_CHECK = """
Episode log for XB4GS9ShBRE must contain:
  '[T7_CENTROID_REG_SNAP] upstair px=(...) navigable=False snapped to (...)'
  at a step before any [T6_STAIR_CLIMB_EVAL] output, AND
  'Reach_stair_centroid: True' at some subsequent step (Phase 1 succeeded).
If log shows PASS (centroid already navigable) but still stalls, root cause is
not the centroid pixel itself but Phase 1 PointNav geometry — different fix needed.
If log shows snap fires but consecutive Reach_stair_centroid: False persists,
the snapped pixel is also unreachable (increase radius or use BFS from robot).
"""
