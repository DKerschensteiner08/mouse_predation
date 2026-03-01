# predation_mouse

Python tools for analyzing and manually correcting single-camera mouse prey-capture tracking data (DeepLabCut-based), with summary outputs, plots, and annotated videos.

## Repository Contents

- `hunting_analysis_functions.py`: Core analysis utilities for preprocessing, kinematics, behavioral segmentation, arena geometry, summaries, plotting, and video annotation.
- `hunting_analysis_script.py`: End-to-end GUI-driven pipeline that loads a video and analysis CSV, asks for arena corners and analysis frame window, and writes all outputs.
- `manual_label_corrector.py`: PyQt-based manual correction tool for frame-by-frame label edits and export to `_pythonAnalysis.csv`.
- `hunting_traces.ipynb`, `hunting_traces_video.ipynb`: Notebook workflows for trajectory visualization.
- `test_mouse/`: Example processed outputs and sample test dataset.

## Features

- DeepLabCut data handling (`.h5` and analysis CSV workflows).
- Automatic derivation of midpoint/head position and prey-relative azimuth angles.
- Approach/contact state detection and smoothing.
- Arena border-distance metrics and spatial density distributions.
- Trial-level summaries and publication-ready plots.
- Annotated output video generation.

## Requirements

- Python 3.10+ (recommended)
- `numpy`
- `pandas`
- `matplotlib`
- `scipy`
- `opencv-python`
- `PyQt5`
- `tkinter` (usually included with standard Python on macOS/Linux)

Install dependencies:

```bash
pip install numpy pandas matplotlib scipy opencv-python PyQt5
```

## Typical Workflow

1. (Optional) Manually correct tracking labels:

```bash
python manual_label_corrector.py
```

2. Run the end-to-end hunt analysis:

```bash
python hunting_analysis_script.py
```

3. In the GUI prompts:
- Select `.mp4` video.
- Select associated DLC `.h5`.
- Select analysis CSV (for example `_pythonAnalysis.csv`).
- Choose analysis `start_frame` and `capture_frame`.
- Click arena corners in order: top-left, top-right, bottom-left, bottom-right.

4. Review generated outputs saved beside the input video/DLC files:
- `*_analysis_full.csv`
- `*_summary.csv`
- `*_distribution.csv`
- `*_density.csv`
- `*_azimuth.csv`
- `*_hunt.png`
- `*_approaches.png`
- `*_speed_and_distance.png`
- `*_azimuth_approach_only.png`
- `*_annotated_slow.avi`

## Notes

- `example_mouse/` is intentionally excluded from version control in this repo.
- Analysis scripts use GUI backends (`TkAgg` / PyQt), so they are best run in a desktop environment.
