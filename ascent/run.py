import os
import subprocess
import sys

_conda_prefix = os.environ.get("CONDA_PREFIX", "/home/teeshan/miniconda3/envs/habitat_clean")
_torch_lib = subprocess.check_output(
    [sys.executable, "-c",
     "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"],
    text=True,
).strip()

os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = "/tmp/10_nvidia_535_288_01.json"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
os.environ["EGL_PLATFORM"] = "device"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["MAGNUM_GPU_DEVICE"] = "0"
os.environ["HABITAT_ENV_DEBUG"] = "1"
os.environ["LD_LIBRARY_PATH"] = (
    f"{_conda_prefix}/lib:/usr/lib/x86_64-linux-gnu:{_torch_lib}"
)
os.environ["LD_PRELOAD"] = f"{_conda_prefix}/lib/libstdc++.so.6"
os.environ.pop("DISPLAY", None)

# Prewarm Habitat-Sim EGL before torch/lavis/open3d/etc touch CUDA/GL.
try:
    import habitat_sim

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = "NONE"
    sim_cfg.gpu_device_id = 0

    sensor = habitat_sim.CameraSensorSpec()
    sensor.uuid = "color_sensor"
    sensor.sensor_type = habitat_sim.SensorType.COLOR
    sensor.resolution = [16, 16]

    agent_cfg = habitat_sim.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor]

    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)
    sim.close()
    print("DEBUG: prewarmed Habitat-Sim EGL")
except Exception as e:
    print("DEBUG: Habitat-Sim prewarm failed:", repr(e))


import hydra  # noqa

from habitat import get_config  # noqa
from habitat.config import read_write
from habitat.config.default import patch_config
from habitat.config.default_structured_configs import register_hydra_plugin
from habitat_baselines.run import execute_exp
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin
from omegaconf import DictConfig
from omegaconf import OmegaConf
import sys
sys.path.insert(0, "third_party/frontier_exploration")
sys.path.insert(0, "third_party/depth_camera_filtering")
sys.path.insert(0, "third_party/vlfm")
import vlfm.measurements.traveled_stairs  # noqa: F401
import vlfm.obs_transformers.resize  # noqa: F401
import vlfm.policy.action_replay_policy  # noqa: F401
import vlfm.policy.habitat_policies  # noqa: F401
from habitat_baselines.config.default_structured_configs import (
    HabitatBaselinesConfigPlugin,
)

from ascent import ascent_policy
from ascent import ascent_trainer 

class HabitatConfigPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(provider="ascent", path="config/")


# 只注册一次，在这里
register_hydra_plugin(HabitatConfigPlugin)

@hydra.main(
    version_base=None,
    config_path="../experiments",  
    config_name="eval_ascent_hm3d.yaml",  # 修改：文件名不带 .yaml 后缀
)
def main(cfg: DictConfig) -> None:
    cfg = patch_config(cfg)
    execute_exp(cfg, "eval" if cfg.habitat_baselines.evaluate else "train")


if __name__ == "__main__":
    register_hydra_plugin(HabitatBaselinesConfigPlugin)  # 如果需要的话保留
    main()