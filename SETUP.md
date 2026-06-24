# meta-robots Setup Guide

This repo contains the ASCENT navigation system plus the T1–T8 meta-harness search framework used to iteratively improve it. This guide covers everything from environment setup to running the search loop and reproducing paper results.

---

## Repository Structure

```
meta-robots/
├── ascent/               # Core ASCENT navigation policy
├── model_api/            # VLM inference servers (BLIP2, RAM, DFINE, GroundingDINO, SAM, Qwen)
├── experiments/          # Habitat eval configs
├── scripts/              # VLM server launch scripts
├── statistic_priors/     # Category/floor prior data
├── RedNet/               # Semantic segmentation model
├── environment.yml       # Conda env export (habitat_clean)
├── dummy_policy.pth      # Required dummy PointNav checkpoint
└── tracks/
    ├── t1/               # Meta-harness T1 (baseline)
    ├── t2/               # Meta-harness T2
    ├── ...
    └── t8/               # Meta-harness T8 (latest)
        ├── scripts/      # loop.py, propose.py, run_eval.py, ...
        ├── track8_harness/   # Base harness (SR=40% on val_30_t8)
        └── runs/         # All candidate results, scores, analysis
```

Each track `tN/` follows the same layout. The search loop lives in `tracks/tN/scripts/loop.py` and the proposer in `tracks/tN/scripts/propose.py`.

---

## 1. Environment Setup

### 1.1 Conda Environment

The full environment is exported in `environment.yml` (env name: `habitat_clean`, Python 3.9):

```bash
conda env create -f environment.yml
conda activate habitat_clean
```

If you prefer a manual install, follow the ASCENT setup instructions in `README.md` (habitat-sim 0.3.1, habitat-lab v0.3.1, GroundingDINO, MobileSAM, requirements.txt).

### 1.2 Third-Party Submodules

```bash
git submodule update --init --recursive
cd third_party/habitat-lab && git checkout v0.3.1
pip install -e habitat-lab && pip install -e habitat-baselines
cd ../GroundingDINO && pip install -e . --no-build-isolation --no-dependencies
cd ../MobileSAM && pip install -e .
cd ../..
```

### 1.3 EGL Vendor JSON (headless GPU rendering)

Habitat requires an EGL vendor JSON at a fixed path. Run once after each reboot:

```bash
cp /usr/share/glvnd/egl_vendor.d/10_nvidia.json /tmp/10_nvidia_535_288_01.json
```

---

## 2. Pretrained Weights

Download and place in `pretrained_weights/`:

| Model | Filename | Source |
|-------|----------|--------|
| Qwen2.5-7B-Instruct | `Qwen2.5-7b/` | [HuggingFace](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |
| RAM++ | `ram_plus_swin_large_14m.pth` | [HuggingFace](https://huggingface.co/xinyu1205/recognize-anything-plus-model) |
| GroundingDINO | `groundingdino_swint_ogc.pth` | [GitHub](https://github.com/IDEA-Research/GroundingDINO) |
| D-FINE | `dfine_x_obj2coco.pth` | [GitHub](https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_x_obj2coco.pth) |
| MobileSAM | `mobile_sam.pt` | [GitHub](https://github.com/ChaoningZhang/MobileSAM) |
| RedNet | `rednet_semmap_mp3d_40.pth` | [Google Drive](https://drive.google.com/file/d/1U0dS44DIPZ22nTjw0RfO431zV-lMPcvv) |
| Places365 | `resnet50_places365.pth.tar` | [Download](http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar) |

PointNav weights are already included at `third_party/vlfm/data/pointnav_weights.pth`.

---

## 3. Dataset Setup

Follow [Habitat-Lab's Datasets.md](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md) to download HM3D and MP3D scene + episode datasets. Place them under `data/`:

```
data/
├── datasets/objectnav/hm3d/v1/val/
│   ├── val.json.gz
│   └── content/*.json.gz
└── scene_datasets/hm3d/val/
    └── <scene_id>/*.glb
```

The search splits (`val_30_t8`, `val_200_t8`, etc.) are pre-built in `data/datasets/objectnav/hm3d/v1/` and included in this repo. To rebuild them:

```bash
python tracks/t8/scripts/create_splits.py
```

---

## 4. VLM Servers

The meta-harness requires six VLM inference servers running before eval. Launch all at once:

```bash
conda activate habitat_clean
bash scripts/launch_vlm_servers_ascent.sh
```

This starts tmux sessions for each server on ports:

| Port | Model |
|------|-------|
| 13181 | Qwen2.5-7B (LLM planner) |
| 13182 | BLIP2-ITM (object scoring) |
| 13183 | MobileSAM |
| 13184 | GroundingDINO (detection) |
| 13185 | RAM++ (tagging) |
| 13186 | D-FINE (detection) |

Wait ~60s for all servers to come up before running evals.

---

## 5. VLM Oracle (T8 only)

T8 uses a Qwen2.5-VL visual oracle that critiques failure videos after each eval. Set it up once:

```bash
bash tracks/t8/scripts/setup_vlm_oracle.sh   # creates vlm_oracle conda env, downloads model
screen -dmS vlm_oracle bash tracks/t8/scripts/vlm_oracle_server.sh  # port 13187
```

T7 and earlier tracks do not require the oracle.

---

## 6. Running a Single Evaluation

To evaluate a specific candidate harness on a split:

```bash
conda activate habitat_clean
cd /path/to/ascent_pipeline   # must run from ascent root

python tracks/t8/scripts/run_eval.py \
    --candidate tracks/t8/runs/candidate_10 \
    --split val_30_t8
```

Results written to `tracks/t8/runs/candidate_10/scores.json`.

For the paper baseline (no harness, vanilla ASCENT):

```bash
python tracks/t7/scripts/run_eval.py \
    --candidate tracks/t7/runs/candidate_base_ascent \
    --split val_200_t7
```

---

## 7. Running the Meta-Harness Search Loop

### T8 (latest):

```bash
bash tracks/t8/scripts/launch_t8.sh
```

Monitors: `tail -f /tmp/loop_t8.log`

### T7:

```bash
screen -dmS loop_t7 bash -c "
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate habitat_clean
    cd /path/to/ascent_pipeline
    python -u tracks/t7/scripts/loop.py \
        --split val_30_t7 --max-candidates 50 --patience 8 \
        --promo-split val_200_t7 --promo-threshold 0.05 \
        2>&1 | tee /tmp/loop_t7.log
"
```

The loop runs indefinitely:
1. Calls `propose.py` → Claude generates next hypothesis → writes harness files
2. Validates harness structure
3. Runs Habitat eval on `val_30` split (~2.5h per candidate)
4. Runs VLM oracle analysis on failure videos
5. If new best SR and gain ≥ 5%, runs promotion eval on `val_200` split
6. Stops after `--patience` candidates with no improvement

**Note:** The loop reads `analysis_db.json`, `cluster_db.json`, and `hypothesis_db.json` from the track root to build the proposer context. These are updated after each candidate.

---

## 8. Reproducing Paper Results

### Baseline (vanilla ASCENT, no meta-harness)
Already evaluated — see `tracks/t7/runs/candidate_base_ascent/scores.json`:
- **SR = 55.5%** on `val_200_t7` (200-ep HM3D val)

### T7 Best (candidate_10)
Already evaluated — see `tracks/t7/runs/candidate_10/scores.json`:
- **SR = 59.5%** on `val_200_t7` (+4pp over baseline)

### Full 2000-ep HM3D val eval
To run candidate_10 on the complete 2000-episode val set:

```bash
# Build complement splits (val minus val_200_t7, sharded into 4 × 450 eps)
python tracks/t7/scripts/create_val1800_splits.py

# Launch 4 parallel evals (~22h on one H100)
bash tracks/t7/scripts/launch_val1800.sh

# Combine with existing 200-ep result
python tracks/t7/scripts/combine_val2000.py
```

---

## 9. Understanding Track Results

Each `tracks/tN/runs/candidate_K/` contains:

| File | Contents |
|------|----------|
| `harness/` | The actual Python files that monkey-patch ASCENT |
| `harness/meta.py` | Hypothesis metadata (MECHANISM, TARGET_FAILURE_CLASSES, etc.) |
| `scores.json` | SR, SPL, DTG metrics |
| `failure_classification.json` | Per-episode failure type labels |
| `behavioral_fingerprint.json` | Per-scene action sequence hashes |
| `telemetry.jsonl` | Per-step agent state logs |
| `val_30_tN.log` | Raw Habitat stdout |

Track-level files:

| File | Contents |
|------|----------|
| `analysis_db.json` | VLM oracle critiques per scene per candidate |
| `cluster_db.json` | Failure clusters + forbidden moves |
| `hypothesis_db.json` | All past hypotheses (for deduplication) |
| `runs/search_log.jsonl` | SR history across all candidates |

---

## 10. Troubleshooting

**EGL / rendering errors:** Re-run `cp /usr/share/glvnd/egl_vendor.d/10_nvidia.json /tmp/10_nvidia_535_288_01.json`

**VLM server not responding:** Check `tmux ls`, ensure all 6 servers are up before starting eval.

**`promo_baseline_sr` never set:** If the loop was restarted from a non-zero candidate (no `candidate_0`), edit `loop.py` to hardcode `promo_baseline_sr` from the known baseline SR (e.g. `0.40` for T8).

**GPU OOM:** Each eval uses ~2GB CUDA + EGL overhead. Running 4 parallel evals on a single 80GB GPU is safe alongside the VLM servers (~12GB).
