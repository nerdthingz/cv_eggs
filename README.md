# Egg segmentation and tracking (no deep learning in the main algorithm)

| File | What it does | Output |
|---|---|---|
| `run_sam.py` | Task 1: initial SAM 3 run on all images (text prompt "egg") | `sam3_out/` (masks, union masks, overlays, summary.csv) |
| `egg_algo.py` | Tasks 2 and 3: classical egg algorithm, dev/eval evaluation, plots | metrics, comparison images, plots (zipped) |
| `vid_detect.py` | Task 4: detect and track eggs in the video | `vid_result/` (tracked video) |
| `EXPERIMENTS_AND_APPROACH.md` | approach and experiment log | |

## Setup
```
pip install opencv-python numpy pandas matplotlib joblib scipy transformers accelerate huggingface_hub
```
`run_sam.py` needs a Hugging Face token with access to `facebook/sam3` (gated model). The scripts use Kaggle paths at the top; change them to run elsewhere.

## Run (in this order)
1. `python run_sam.py`
2. `python egg_algo.py`
3. `python vid_detect.py`

Tracked video: _add Google Drive link here_
