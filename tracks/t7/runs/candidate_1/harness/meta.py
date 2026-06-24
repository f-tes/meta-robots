TARGET_FAILURE_CLASSES = ["navmesh_disconnected_stair_centroid"]
TARGET_SCENES = ["q3zU7Yy5E5s"]

HYPOTHESIS = (
    "The upstair centroid at [-2.12027027, 3.27567568] in q3zU7Yy5E5s lies on a "
    "navigable_map pixel that belongs to a disconnected island component, physically "
    "isolated from the robot's reachable floor region. All gcts_streak-based "
    "interventions (any threshold), the ±50px box navmesh check at gcts_step=0 "
    "(T6 c5), and stair.py check_upstair_navmesh() at streak==1 (T6 c9/c10) are "
    "ruled out. The ±50px box fails because adjacent pixels from the main floor "
    "component fall within the box, causing a false-pass despite the centroid being "
    "unreachable. The fix must (a) use BFS flood-fill from robot position to test "
    "actual connectivity, not pixel presence, and (b) fire at gcts MODE ENTRY — "
    "the first call to _get_close_to_stair for each new stair centroid target — "
    "which is earlier than all previously tried injection points."
)

MECHANISM = (
    "In patch.py Fix 4, patch Ascent_Policy._get_close_to_stair. Track a per-env "
    "_bfs_checked dict keyed by centroid_tuple. On the first gcts call for each new "
    "centroid: compute robot_rc and centroid_rc in navigable_map [row,col] space "
    "(note: _xy_to_px returns [col,row] per obstacle_map.py:339), run BFS "
    "flood-fill with 4-connectivity on om._navigable_map. If centroid_rc is NOT "
    "reachable from robot_rc, call mc._disable_stair_and_reset_state() and return "
    "_explore immediately — zero gcts PointNav steps wasted. The incumbent "
    "streak>=10 safety net is preserved as a fallback for navmesh gaps discovered "
    "later (e.g., after partial approach)."
)

PREDICTED_CHANGE = (
    "q3zU7Yy5E5s: BFS detects disconnected upstair centroid on first gcts call "
    "(step ~70-72). Stair disabled immediately. Agent resumes explore and may find "
    "downstairs path (confirmed reachable in T5 c8). Expected ~76 wasted steps "
    "recovered. Log: '[T7_BFS_MODE_ENTRY] ... NOT reachable ... upstair disabled'."
)

PREDICTED_SR_DELTA = 0.07

WHY_THIS_WILL_WORK = (
    "q3zU7Yy5E5s burns ~76 steps in the gcts loop on PointNav failures to a "
    "provably unreachable centroid. BFS flood-fill on the 2D navigable_map is the "
    "correct algorithm for connected-component membership: if BFS from robot_px "
    "cannot reach centroid_px, PointNav will never succeed regardless of carrot "
    "distance, snap radius, or attempt count. No previous T7 attempt has combined "
    "(1) BFS connectivity (not pixel value or box scan) with (2) injection before "
    "any gcts PointNav steps. The T6 c5 byte-identical result proves the ±50px box "
    "check passes incorrectly; BFS would return False for the same centroid because "
    "the flood-fill cannot cross the navmesh gap between the main floor and the "
    "disconnected island. Firing at first gcts call (not streak==10) recovers all "
    "~76 wasted steps, not just the last ~46."
)

WHY_ALTERNATIVES_REJECTED = (
    "gcts_streak-based approaches (any threshold N<=10) are explicitly ruled out. "
    "±50px box check (T6 c5) is ruled out — produces byte-identical result because "
    "adjacent main-floor pixels fall within the box. stair.py navmesh checks at "
    "streak==1 (T6 c9/c10) are covered by the gcts_streak ruling. DP levers "
    "(DP9, DP12, DP2, DP3, DP5, DP6, DP7) are all ruled out. "
    "strict_single_pixel_centroid_check would also fail because the centroid pixel "
    "IS navigable (value=1) — it is the connectivity that is broken, not the pixel "
    "value. Only BFS from robot position correctly identifies the disconnection."
)

FALSIFIABILITY_CHECK = (
    "Log line '[T7_BFS_MODE_ENTRY] ... NOT reachable ... upstair disabled' must "
    "appear in q3zU7Yy5E5s episodes. Episode must NOT subsequently log any gcts "
    "PointNav actions for that stair. Step count at stair-disable must be "
    "significantly lower than the ~76-step gcts burn seen in candidate_0."
)
