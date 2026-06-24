"""
Loads the active harness instance.

Track 2: ASCENT_PIPELINE_HARNESS_PATH → PipelineHarness
Track 3: ASCENT_T3_HARNESS_PATH       → Track3Harness  (takes priority)
Track 4: ASCENT_T4_HARNESS_PATH       → Track4Harness  (takes priority over T3)
Track 5: ASCENT_T5_HARNESS_PATH       → Track5Harness  (takes priority over T4)
Track 6: ASCENT_T6_HARNESS_PATH       → Track6Harness  (takes priority over all;
                                         points to a harness/ directory, not a .py file)
Track 7: ASCENT_T7_HARNESS_PATH       → Track7Harness  (takes priority over all;
                                         points to a harness/ directory, not a .py file)

All ASCENT decision points call get_harness().<method>(...).
get_harness().apply() is called once at startup for structural patches.
"""
import importlib.util
import os
import sys

_harness = None
_harness_path = None


def _load_directory_harness(harness_dir: str, class_name: str):
    """Load a directory-based harness package (Track 5+)."""
    parent = os.path.dirname(os.path.abspath(harness_dir))
    pkg_name = os.path.basename(os.path.abspath(harness_dir))
    init_path = os.path.join(harness_dir, "__init__.py")

    if parent not in sys.path:
        sys.path.insert(0, parent)

    # Remove stale cached modules so reload picks up changes.
    stale = [k for k in sys.modules if k == pkg_name or k.startswith(pkg_name + ".")]
    for k in stale:
        del sys.modules[k]

    spec = importlib.util.spec_from_file_location(
        pkg_name,
        init_path,
        submodule_search_locations=[harness_dir],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, class_name)()


def get_harness():
    global _harness, _harness_path
    # Priority: T8 > T7 > T6 > T5 > T4 > T3 > T2
    is_dir = False
    t8_path = os.environ.get("ASCENT_T8_HARNESS_PATH")
    t7_path = os.environ.get("ASCENT_T7_HARNESS_PATH")
    t6_path = os.environ.get("ASCENT_T6_HARNESS_PATH")
    t5_path = os.environ.get("ASCENT_T5_HARNESS_PATH")
    t4_path = os.environ.get("ASCENT_T4_HARNESS_PATH")
    t3_path = os.environ.get("ASCENT_T3_HARNESS_PATH")
    if t8_path:
        path = t8_path
        class_name = "Track8Harness"
        is_dir = True
    elif t7_path:
        path = t7_path
        class_name = "Track7Harness"
        is_dir = True
    elif t6_path:
        path = t6_path
        class_name = "Track6Harness"
        is_dir = True
    elif t5_path:
        path = t5_path
        class_name = "Track5Harness"
        is_dir = True
    elif t4_path:
        path = t4_path
        class_name = "Track4Harness"
    elif t3_path:
        path = t3_path
        class_name = "Track3Harness"
    else:
        path = os.environ.get(
            "ASCENT_PIPELINE_HARNESS_PATH",
            "/home/teeshan/meta_harness_pipeline/pipeline_harness.py",
        )
        class_name = "PipelineHarness"

    if _harness is None or path != _harness_path:
        if is_dir:
            _harness = _load_directory_harness(path, class_name)
        else:
            spec = importlib.util.spec_from_file_location("harness_module", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _harness = getattr(mod, class_name)()
        _harness.apply()
        _harness_path = path
    return _harness


def reset_harness():
    """Force reload on next get_harness() call."""
    global _harness, _harness_path
    _harness = None
    _harness_path = None


def safe_emit(method_name, *args, **kwargs):
    """Call a harness method only if it exists — no-op otherwise. For T4 telemetry hooks."""
    try:
        h = get_harness()
        fn = getattr(h, method_name, None)
        if fn is not None:
            fn(*args, **kwargs)
    except Exception:
        pass
