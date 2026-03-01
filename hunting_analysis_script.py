# -*- coding: utf-8 -*-
"""
@author: @Daniel Kerschensteiner

End-to-end hunt analysis & annotation driver (single camera).
Loads a precomputed per-frame Analysis CSV, asks you to mark the arena on the
video, converts pixels→cm via a projective transform, derives kinematics/angles,
segments approach/contact, computes arena-border metrics and distributions,
plots figures, annotates the video, and saves all outputs next to the source files.

Key Steps:
-Pick files (GUI): .mp4 video → DLC .h5 (used for naming) → _Analysis.csv (framewise tracks).
-Set geometry: click arena corners on a video frame (TL→TR→BL→BR) → cm/pixel estimate.
-Bound trial: enter START and CAPTURE frames; analyses are restricted to [start_frame, capture_frame).
-Compute signals: mid/head, head/body azimuth to cricket; distance mid↔cricket (cm);
mouse/cricket speed (+acceleration), approaches/contacts with smoothing.
-Project to arena (cm): affine_transform() to get madj_x/y, cadj_x/y; then arena-relative
turning (head/body), distances to borders, and path-to-border; corners stored in df.
-Distributions: distance-to-border PDFs, 2D spatial densities (mouse/cricket × states),
and azimuth histograms (optionally limited by distance to prey).
-Plots: full paths, approach-only paths, speed & distance over time (green=approach, red=contact),
and azimuth histogram.
-Video: writes an annotated AVI (MJPG) with time, state banners, keypoints, azimuth line, and stats.

Important Logic:
-Approach is suppressed during contact (contact takes precedence).
-Most readers accept to_capture_time to clip at the first ‘captured’ frame; now also trimmed by start_frame.
-All plotting uses safe copies to avoid chained-assignment warnings.

Defaults (editable at top):
-FPS=30; contact_distance=4 cm; speed_threshold=10 cm/s; diff_frames=4; diff_speed=−20 cm/s (scaled by FPS);
body_azimuth gate=60°.
-Arena size: 45 × 38 cm (x_size, y_size). Dist bins: 0…19 cm (bin_size=1). Density grid: 20×20 within [[0,38],[0,45]].
-Azimuth bins: −180…180° in 5° steps; max_dist_azimuth=5 cm.
-Video annotation: fps_out=10, border pad=40 px; draws keypoints/lines and state labels.

Inputs (expected columns in _Analysis.csv):
-nose_x/y, l_ear_x/y, r_ear_x/y, tail_base_x/y, cricket_x/y, time (s), etc. (mid_x/y will be computed if absent).

Outputs (saved beside the .h5 base name):
-<base>_analysis_full.csv (augmented per-frame data incl. madj/cadj & states)
-<base>_summary.csv (one-row trial summary: counts, latencies, times, probabilities)
-<base>_distribution.csv (distance-to-border PDFs)
-<base>_density.csv (2D spatial densities, flattened)
-<base>_azimuth.csv (state-wise azimuth PDFs)
-Figures: <video>_hunt.png, _approaches.png, _speed_and_distance.png, _azimuth_approach_only.png
-Video: <video>_annotated_slow.avi

Notes:
-Requires hunting_analysis_functions on PYTHONPATH.
-Working directory is set to the video’s folder; close the corner-selection window after 4 clicks.
-Use the same corner order (TL→TR→BL→BR) consistently for accurate transforms.
"""

import pandas as pd
import numpy as np
import tkinter
from tkinter import filedialog, Tk, simpledialog, messagebox
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2

from hunting_analysis_functions import (
    affine_transform,
    add_corners,
    annotate_video,
    calculate_head,
    calculate_mid,
    get_approaches,
    get_azimuth_body,
    get_azimuth_body_arena,
    get_azimuth_head,
    get_azimuth_head_arena,
    get_azimuth_hist,
    get_borders,
    get_contacts,
    get_cricket_speed,
    get_density,
    get_distance_path_to_borders,
    get_distance_to_borders,
    get_distance_to_cricket,
    get_distribution,
    get_mouse_acceleration,
    get_mouse_speed,
    get_save_path_csv,
    get_save_path_csv_azimuth,
    get_save_path_csv_density,
    get_save_path_csv_distribution,
    get_save_path_csv_summary,
    pixel_size_from_arena_coordinates,
    plot_approaches,
    plot_azimuth_hist,
    plot_hunt,
    plot_speeds_and_distance,
    select_arena_manual,
    set_start_and_capture_frames,   # NEW
    smooth_approaches,
    smooth_contacts,
    summarize_df
)

# --- Setup ---
root = Tk()
root.withdraw()  # Hide the main tkinter window

# --- User Inputs ---
print("Please select the required files.")
video_path = filedialog.askopenfilename(
    parent=root, title='Choose .mp4 video file')
if not video_path:
    raise Exception("Video file not selected. Exiting.")

h5_file = filedialog.askopenfilename(parent=root, title='Choose DLC .h5 file')
if not h5_file:
    raise Exception("H5 file not selected. Exiting.")

df_path = filedialog.askopenfilename(
    parent=root, title='Choose _Analysis.csv file to process')
if not df_path:
    raise Exception("Analysis CSV file not selected. Exiting.")

os.chdir(os.path.dirname(os.path.abspath(video_path)))

# --- Load Data ---
print(f"Loading data from {os.path.basename(df_path)}...")
df = pd.read_csv(df_path)
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)

# --- Determine frame count for defaults ---
frame_count = None
try:
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Could not open video for frame count.")
    frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    vid.release()
except Exception as e:
    print(f"Warning: {e}. Falling back to CSV length for frame count.")
    frame_count = len(df)

# --- Ask for START and CAPTURE frames (CAPTURE is exclusive end) ---
# Defaults: start=0, capture=frame_count (use frame_count to include the last frame)
start_default = 0
capture_default = frame_count

start_frame = simpledialog.askinteger(
    "Start frame",
    f"Enter START frame (0 .. {max(0, frame_count-1)}):",
    initialvalue=start_default,
    minvalue=0,
    maxvalue=max(0, frame_count-1),
    parent=root
)
if start_frame is None:
    start_frame = start_default

capture_frame = simpledialog.askinteger(
    "Capture frame (exclusive end)",
    f"Enter CAPTURE frame (exclusive end; use {frame_count} to include last frame):",
    initialvalue=capture_default,
    minvalue=0,
    maxvalue=frame_count,
    parent=root
)
if capture_frame is None:
    capture_frame = capture_default

# Sanity checks
if capture_frame < start_frame:
    messagebox.showwarning(
        "Adjusting CAPTURE", "Capture frame was before start; adjusting to match start.")
    capture_frame = start_frame

# Persist start/capture into the DataFrame
df = set_start_and_capture_frames(
    df, start_frame=start_frame, capture_frame=capture_frame)
print(
    f"Analysis window: start_frame={start_frame}, capture_frame={capture_frame} (exclusive)")

# --- Parameters ---
frame_rate = 30
select_frame = 10
speed_threshold = 10
contact_distance = 6  # 6 cm for dunnarts, 4 cm for mice
windowSize = 8
diff_frames = 4
diff_speed = -20
body_azimuth = 60
max_dist = 19
bin_size = 1
y_size = 38
x_size = 45
bin_num = 20
bin_range = [[0, y_size], [0, x_size]]
azimuth_bin_size = 5
azimuth_range = [-180, 180]
max_dist_azimuth = 5

# --- Main Processing Pipeline ---

# 1. Manual Inputs: arena corners and cm/px
corners = select_arena_manual(video_path, frame_number=select_frame)
cm_per_pixel = pixel_size_from_arena_coordinates(
    corners, arena_size_x=x_size, arena_size_y=y_size)
print(f"Calculated cm/pixel: {cm_per_pixel:.4f}")

# 2. Compute analysis parameters
print("Computing analysis parameters...")
# Midpoint if missing
if 'mid_x' not in df.columns or 'mid_y' not in df.columns:
    print("Midpoint columns not found. Calculating now...")
    df = calculate_mid(df)
else:
    print("Midpoint columns already exist. Skipping calculation.")

df = calculate_head(df)
df = get_azimuth_head(df)
df = get_azimuth_body(df)
df = get_distance_to_cricket(df, cm_per_pixel)
df = get_mouse_speed(df, frame_rate, cm_per_pixel,
                     smooth_frames=15, smooth_order=3)
df = get_cricket_speed(df, frame_rate, cm_per_pixel,
                       smooth_frames=15, smooth_order=3)
df = get_mouse_acceleration(df, smooth_frames=15, smooth_order=3)
df = get_contacts(df, contact_distance)
df = smooth_contacts(df, windowSize=int(windowSize/2))
df = get_approaches(df, speed_threshold, diff_frames=diff_frames,
                    diff_speed=diff_speed, frame_rate=frame_rate, body_azimuth=body_azimuth)
df = smooth_approaches(df, windowSize=windowSize)

# No approach while contact (applied globally; windowing happens in downstream readers)
df.loc[df['contact'] == 1, 'approach'] = 0
print("Analysis parameters computed.")

# 3. Projective transformation
print("Applying projective transformation...")
target_corners = np.array([[0, 0], [x_size, 0], [0, y_size], [x_size, y_size]])

mouse_points = np.array((df['mid_x'], df['mid_y']))
df[['madj_x', 'madj_y']] = affine_transform(
    corners, target_corners, mouse_points)

cricket_points = np.array((df['cricket_x'], df['cricket_y']))
df[['cadj_x', 'cadj_y']] = affine_transform(
    corners, target_corners, cricket_points)

# 4. Compute arena-relative parameters
df = get_azimuth_head_arena(df)
df = get_azimuth_body_arena(df)
borders = get_borders(target_corners, pts_per_border=1000)
df = get_distance_to_borders(df, borders)
df = get_distance_path_to_borders(df, borders, n_samples=100)
df = add_corners(df, corners)
print("Arena-relative parameters computed.")

# 5. Compute distributions, densities, and histograms (windowed)
print("Computing distributions and histograms...")
dist_bins = np.arange(0, max_dist + bin_size, bin_size)
# now also trimmed by start_frame internally
df_distribution = get_distribution(df, dist_bins, to_capture_time=True)

df_density = get_density(df, bin_num, bin_range, to_capture_time=True)

num_bins = int((azimuth_range[1] - azimuth_range[0]) / azimuth_bin_size) + 1
azimuth_bins = np.linspace(azimuth_range[0], azimuth_range[1], num_bins)
df_azimuth = get_azimuth_hist(
    df, azimuth_bins, max_dist_azimuth, to_capture_time=True)
print("Distributions computed.")

# 6. Plot results and save figures (windowed)
print("Generating plots...")
plot_hunt(df, to_capture_time=True, video_path=video_path, save_fig=True)
plot_approaches(df, to_capture_time=True, video_path=video_path, save_fig=True)
plot_speeds_and_distance(df, mouse=True, cricket=True, to_capture_time=True,
                         contact_distance=contact_distance, video_path=video_path, save_fig=True)
plot_azimuth_hist(df, n_bins=20, approach_only=True,
                  video_path=video_path, save_fig=True)
print("Plots saved.")

# 7. Annotate video (windowed)
print("Annotating video...")
annotate_video(
    df,
    video_path,
    fps_out=10,
    to_capture_time=True,
    show_time=True,
    show_borders=True,
    label_bodyparts=True,
    show_approaches=True,
    show_approach_number=True,
    show_contacts=True,
    show_contact_number=True,
    show_azimuth=True,
    show_azimuth_lines=True,
    save_video=True,
    show_video=False,
    border_size=40,
    show_distance=True,
    show_speed=True,
    video_path_ext='_annotated_slow',
    transform_to_arena=False,
    arena_width_cm=x_size,
    arena_height_cm=y_size
)
print("Video annotation complete.")

# 8. Summarize and save DataFrames (window-aware)
print("Summarizing and saving data...")
trial_id = os.path.basename(video_path).split('.')[0]
try:
    condition = trial_id.split('_')[0]
except Exception:
    condition = "WT"
summary_df = summarize_df(df, trial_id=trial_id, condition=condition)

# Get save paths programmatically
save_path_full = get_save_path_csv(h5_file)
save_path_summary = get_save_path_csv_summary(h5_file)
save_path_distribution = get_save_path_csv_distribution(h5_file)
save_path_density = get_save_path_csv_density(h5_file)
save_path_azimuth = get_save_path_csv_azimuth(h5_file)

# Save all data
df.to_csv(save_path_full, index=False)
summary_df.to_csv(save_path_summary, index=False)
df_distribution.to_csv(save_path_distribution, index=False)
df_density.to_csv(save_path_density, index=False)
df_azimuth.to_csv(save_path_azimuth, index=False)

root.destroy()
print("\nAnalysis complete. All files saved.")
