"""
patch.py — NULL patch for base ASCENT comparison candidate.

apply() is intentionally a no-op. This gives vanilla ASCENT behaviour:
no Fix 0 (KeyError guard), no Fix 1 (no-quit rescue), no Fix 2 (centroid
bypass), no Fix 3 (double floor re-init guard), no Fix 4 (early gcts disable).
Combined with DP9 restoring the 0.8m carrot, this is unmodified ASCENT.
"""


class PatchMixin:
    def apply(self) -> None:
        """SDP-A: No-op. Vanilla ASCENT — no monkey-patches applied."""
        pass
