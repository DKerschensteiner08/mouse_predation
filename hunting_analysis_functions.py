# -*- coding: utf-8 -*-
"""
@author: @Daniel Kerschensteiner

Analysis helpers for mouse–cricket hunts from single-camera DeepLabCut (.h5/.csv).
Converts DLC tracks to tidy frames, cleans (likelihood gate → interpolation → Savitzky–Golay),
derives kinematics/angles, segments approach/contact, computes border metrics, summarizes trials,
and optionally annotates video.

Key Features:
-Robust ingest/cleaning: h5_to_df(frame_rate), interpolate_unlikely_label_positions(...), smooth_labels(...).
-Core geometry: calculate_mid(), calculate_head(); distance mid↔cricket (cm), mouse/cricket speed, acceleration.
-Angles (deg): signed head/body azimuth toward cricket; arena turning (frame-to-frame heading change).
-Behavioral epochs: get_contacts(contact_distance), get_approaches(...); smooth_*; set_start_and_capture_frames() to bound analyses.
-Border geometry: select_arena_manual() → pixel_size_from_arena_coordinates() → get_borders() → affine_transform()/add_corners();
distances to borders and path-to-border.
-Distributions: distance-to-border PDFs, 2D spatial densities, and azimuth histograms with distance cutoffs.
-Summaries/plots/output: summarize_df(trial_id, condition); plot_hunt(), plot_approaches(),
plot_speeds_and_distance(), plot_azimuth_hist(); annotate_video(...); save-path helpers for CSVs.

Workflow (typical):
-Import DLC → clean → calculate_mid/head → kinematics (cm_per_pixel) → angles → events → set_start_and_capture_frames() → metrics/plots/summary.
-For border analyses/trajectory plots, provide adjusted coordinates ('madj_','cadj_') via your perspective mapping.

Defaults (overridable):
-FPS=30; cm/px=0.23; contact_distance=4 cm; Savitzky–Golay window=15, order=3.
-Approach gate: speed>10 cm/s, |body azimuth|<60°, closing rate<20 cm/s over 4 frames (scaled by FPS).

Notes:
-Angles are signed (left negative, right positive). Arena azimuths are heading changes (deg).
-'approach' and 'contact' are binary vectors; most functions accept to_capture_time and now also respect start_frame.
-NaN-safe arithmetic (clipped arccos, interpolation) and copy-on-plot to avoid chained-assignment warnings.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os
import cv2
import math

# ------------------------
# Helpers for analysis window
# ------------------------

def set_start_and_capture_frames(df, start_frame=0, capture_frame=None):
    """
    Adds 'started' and 'captured' columns to bound analyses.
    - 'started' == 1 from start_frame to end, else 0.
    - 'captured' == 1 from capture_frame to end, else 0.
      If capture_frame is None or >= len(df), no frames are marked captured (full length).
    """
    n = len(df)
    # clamp
    if start_frame is None:
        start_frame = 0
    start_frame = max(0, min(int(start_frame), max(0, n - 1)))
    if capture_frame is None:
        capture_frame = n
    capture_frame = max(0, min(int(capture_frame), n))  # exclusive end allowed (=n)

    df['started'] = 0
    if start_frame < n:
        df.loc[start_frame:, 'started'] = 1

    df['captured'] = 0
    if capture_frame < n:
        df.loc[capture_frame:, 'captured'] = 1

    # Optional: store values for reference
    df['start_frame_value'] = start_frame
    df['capture_frame_value'] = capture_frame
    print(f"Window set: start_frame={start_frame}, capture_frame={capture_frame} (exclusive).")
    return df


def set_capture_frame(df, capture_frame):
    """(Back-compat) Sets only the capture frame as before."""
    return set_start_and_capture_frames(df, start_frame=0, capture_frame=capture_frame)


def get_capture_frame(df):
    """Returns the index of the first frame marked as 'captured' (exclusive end)."""
    try:
        if 1 not in df.get('captured', pd.Series(0, index=df.index)).values:
            return len(df)
        capture_idx = df['captured'].idxmax()
        if df.loc[capture_idx, 'captured'] == 1:
            return capture_idx
        return len(df)
    except (KeyError, TypeError):
        return len(df)


def get_start_frame(df):
    """Returns the index of the analysis start frame (inclusive)."""
    try:
        if 1 in df.get('started', pd.Series(0, index=df.index)).values:
            start_idx = df['started'].idxmax()
            if df.loc[start_idx, 'started'] == 1:
                return start_idx
    except (KeyError, TypeError):
        pass
    # Fallback to stored scalar if present
    try:
        return int(df['start_frame_value'].iloc[0])
    except Exception:
        return 0


def _slice_by_window(df, from_start_time=False, to_capture_time=False):
    """
    Returns a copy of df restricted to [start_frame, capture_frame) if requested.
    Keeps original indices (no reset) so absolute indexing still works.
    """
    start_idx = get_start_frame(df) if from_start_time else 0
    end_idx = get_capture_frame(df) if to_capture_time else len(df)

    if end_idx < len(df):
        return df.loc[start_idx:end_idx - 1].copy()
    else:
        return df.loc[start_idx:].copy()

# ------------------------
# Ingest & preprocessing
# ------------------------

def h5_to_df(h5_file, frame_rate=30):
    """Converts a DeepLabCut .h5 file to a pandas DataFrame."""
    pos = pd.read_hdf(h5_file)
    network_name = pos.columns[0][0]

    cricket_x = pos.get((network_name, 'cricket', 'x'), pd.Series(np.nan, index=pos.index)).values
    cricket_y = pos.get((network_name, 'cricket', 'y'), pd.Series(np.nan, index=pos.index)).values
    cricket_likelihood = pos.get((network_name, 'cricket', 'likelihood'), pd.Series(np.nan, index=pos.index)).values
    nose_x = pos.get((network_name, 'nose', 'x'), pd.Series(np.nan, index=pos.index)).values
    nose_y = pos.get((network_name, 'nose', 'y'), pd.Series(np.nan, index=pos.index)).values
    nose_likelihood = pos.get((network_name, 'nose', 'likelihood'), pd.Series(np.nan, index=pos.index)).values
    leftear_x = pos.get((network_name, 'l_ear', 'x'), pd.Series(np.nan, index=pos.index)).values
    leftear_y = pos.get((network_name, 'l_ear', 'y'), pd.Series(np.nan, index=pos.index)).values
    leftear_likelihood = pos.get((network_name, 'l_ear', 'likelihood'), pd.Series(np.nan, index=pos.index)).values
    rightear_x = pos.get((network_name, 'r_ear', 'x'), pd.Series(np.nan, index=pos.index)).values
    rightear_y = pos.get((network_name, 'r_ear', 'y'), pd.Series(np.nan, index=pos.index)).values
    rightear_likelihood = pos.get((network_name, 'r_ear', 'likelihood'), pd.Series(np.nan, index=pos.index)).values
    tailbase_x = pos.get((network_name, 'tail_base', 'x'), pd.Series(np.nan, index=pos.index)).values
    tailbase_y = pos.get((network_name, 'tail_base', 'y'), pd.Series(np.nan, index=pos.index)).values
    tailbase_likelihood = pos.get((network_name, 'tail_base', 'likelihood'), pd.Series(np.nan, index=pos.index)).values

    mid_x = np.mean([leftear_x, rightear_x], axis=0)
    mid_y = np.mean([leftear_y, rightear_y], axis=0)

    n_frames = len(cricket_x)
    frames = np.arange(n_frames)
    time = np.arange(n_frames) / frame_rate

    df = pd.DataFrame({
        'frame_number': frames,
        'time': time,
        'leftear_x': leftear_x,
        'leftear_y': leftear_y,
        'leftear_likelihood': leftear_likelihood,
        'rightear_x': rightear_x,
        'rightear_y': rightear_y,
        'rightear_likelihood': rightear_likelihood,
        'mid_x': mid_x,
        'mid_y': mid_y,
        'nose_x': nose_x,
        'nose_y': nose_y,
        'nose_likelihood': nose_likelihood,
        'tailbase_x': tailbase_x,
        'tailbase_y': tailbase_y,
        'tailbase_likelihood': tailbase_likelihood,
        'cricket_x': cricket_x,
        'cricket_y': cricket_y,
        'cricket_likelihood': cricket_likelihood
    })

    return df

# ------------------------
# Geometry & kinematics
# ------------------------

def calculate_mid(df):
    """Calculates the midpoint between the ears. Vectorized for speed."""
    df['mid_x'] = (df['leftear_x'] + df['rightear_x']) / 2
    df['mid_y'] = (df['leftear_y'] + df['rightear_y']) / 2
    return df


def calculate_head(df):
    """Calculates the centroid of the head."""
    df['head_x'] = df[['leftear_x', 'rightear_x', 'nose_x']].mean(axis=1)
    df['head_y'] = df[['leftear_y', 'rightear_y', 'nose_y']].mean(axis=1)
    print('head centroid position columns [head_x and head_y] added')
    return df


def interpolate_unlikely_label_positions(df, likelihood_cutoff=0.9, cricket=True, nose=True, tailbase=True):
    """Interpolates positions for labels with low likelihood scores."""
    if cricket:
        df.loc[df['cricket_likelihood'] < likelihood_cutoff, ['cricket_x', 'cricket_y']] = np.nan
        df[['cricket_x', 'cricket_y']] = df[['cricket_x', 'cricket_y']].interpolate(method='linear', limit_direction='both')
        print('unlikely cricket label positions linearly interpolated')

    if nose:
        df.loc[df['nose_likelihood'] < likelihood_cutoff, ['nose_x', 'nose_y']] = np.nan
        df[['nose_x', 'nose_y']] = df[['nose_x', 'nose_y']].interpolate(method='linear', limit_direction='both')
        print('unlikely nose label positions linearly interpolated')

    if tailbase:
        df.loc[df['tailbase_likelihood'] < likelihood_cutoff, ['tailbase_x', 'tailbase_y']] = np.nan
        df[['tailbase_x', 'tailbase_y']] = df[['tailbase_x', 'tailbase_y']].interpolate(method='linear', limit_direction='both')
        print('unlikely tailbase label positions linearly interpolated')

    return df


def smooth_labels(df, smooth_frames=15, smooth_order=3):
    """Applies a Savitzky-Golay filter to smooth label positions."""
    cols_to_smooth = ['leftear_x', 'leftear_y', 'rightear_x', 'rightear_y', 'nose_x', 'nose_y', 'tailbase_x', 'tailbase_y', 'cricket_x', 'cricket_y']
    df[cols_to_smooth] = savgol_filter(df[cols_to_smooth], smooth_frames, smooth_order, axis=0)
    print('label positions smoothed')
    return df


def lineardistance(x1, x2, y1, y2):
    """Calculates the Euclidean distance between two points."""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def get_distance_to_borders(df, borders):
    """Calculates the minimum distance from mouse and cricket to arena borders."""
    cricket_to_borders = np.zeros(len(df))
    mouse_to_borders = np.zeros(len(df))

    for i in range(len(df)):
        cricket_to_borders[i] = np.min(lineardistance(df.loc[i, 'cadj_x'], borders[:, 0], df.loc[i, 'cadj_y'], borders[:, 1]))
        mouse_to_borders[i] = np.min(lineardistance(df.loc[i, 'madj_x'], borders[:, 0], df.loc[i, 'madj_y'], borders[:, 1]))

    df['cricket_to_borders'] = cricket_to_borders
    df['mouse_to_borders'] = mouse_to_borders
    print('mouse_to_borders and cricket_to_borders columns added')
    return df


def get_distance_path_to_borders(df, borders, n_samples=100):
    """Calculates the mean distance of the direct path between mouse and cricket to the borders."""
    path_to_borders = np.zeros(len(df))

    for i in range(len(df)):
        path_x = np.linspace(df.loc[i, 'madj_x'], df.loc[i, 'cadj_x'], n_samples)
        path_y = np.linspace(df.loc[i, 'madj_y'], df.loc[i, 'cadj_y'], n_samples)
        path_distance = np.zeros(n_samples)

        for j in range(n_samples):
            path_distance[j] = np.min(lineardistance(path_x[j], borders[:, 0], path_y[j], borders[:, 1]))

        path_to_borders[i] = np.mean(path_distance)

    df['path_to_borders'] = path_to_borders
    print('path_to_borders column added')
    return df


def get_azimuth_head(df):
    """Calculates the head azimuth relative to the cricket."""
    a = lineardistance(df['cricket_x'], df['mid_x'], df['cricket_y'], df['mid_y'])
    b = lineardistance(df['cricket_x'], df['nose_x'], df['cricket_y'], df['nose_y'])
    c = lineardistance(df['nose_x'], df['mid_x'], df['nose_y'], df['mid_y'])

    cos_B = np.clip(((c**2) + (a**2) - (b**2)) / (2 * c * a), -1.0, 1.0)
    B = np.degrees(np.arccos(cos_B))

    azimuth = B
    leftear_distance = lineardistance(df['cricket_x'], df['leftear_x'], df['cricket_y'], df['leftear_y'])
    rightear_distance = lineardistance(df['cricket_x'], df['rightear_x'], df['cricket_y'], df['rightear_y'])

    azimuth[leftear_distance < rightear_distance] = -azimuth[leftear_distance < rightear_distance]

    df['azimuth_head'] = azimuth
    print('azimuth_head column added to dataframe')
    return df


def get_azimuth_body(df):
    """Calculates the body azimuth relative to the cricket."""
    a = lineardistance(df['cricket_x'], df['tailbase_x'], df['cricket_y'], df['tailbase_y'])
    b = lineardistance(df['cricket_x'], df['mid_x'], df['cricket_y'], df['mid_y'])
    c = lineardistance(df['mid_x'], df['tailbase_x'], df['mid_y'], df['tailbase_y'])

    cos_B = np.clip(((c**2) + (a**2) - (b**2)) / (2 * c * a), -1.0, 1.0)
    B = np.degrees(np.arccos(cos_B))

    azimuth = B
    leftear_distance = lineardistance(df['cricket_x'], df['leftear_x'], df['cricket_y'], df['leftear_y'])
    rightear_distance = lineardistance(df['cricket_x'], df['rightear_x'], df['cricket_y'], df['rightear_y'])

    azimuth[leftear_distance < rightear_distance] = -azimuth[leftear_distance < rightear_distance]

    df['azimuth_body'] = azimuth
    print('azimuth_body column added to dataframe')
    return df


def get_azimuth_head_arena(df):
    """Calculates the change in head direction relative to the arena (turning speed)."""
    heading_x = (df['nose_x'] - df['mid_x']).values
    heading_y = (df['nose_y'] - df['mid_y']).values

    heading_angle = np.full(len(heading_x), np.nan)

    for i in range(len(heading_x) - 1):
        vec_current = np.array([heading_x[i], heading_y[i]])
        vec_future = np.array([heading_x[i+1], heading_y[i+1]])

        norm_current = np.linalg.norm(vec_current)
        norm_future = np.linalg.norm(vec_future)

        if norm_current > 0 and norm_future > 0:
            cos_angle = np.dot(vec_current, vec_future) / (norm_current * norm_future)
            heading_angle[i] = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    df['azimuth_head_arena'] = heading_angle
    print('azimuth_head_arena column added to dataframe')
    return df


def get_azimuth_body_arena(df):
    """Calculates the change in body direction relative to the arena (turning speed)."""
    bearing_x = (df['mid_x'] - df['tailbase_x']).values
    bearing_y = (df['mid_y'] - df['tailbase_y']).values

    bearing_angle = np.full(len(bearing_x), np.nan)

    for i in range(len(bearing_x) - 1):
        vec_current = np.array([bearing_x[i], bearing_y[i]])
        vec_future = np.array([bearing_x[i+1], bearing_y[i+1]])

        norm_current = np.linalg.norm(vec_current)
        norm_future = np.linalg.norm(vec_future)

        if norm_current > 0 and norm_future > 0:
            cos_angle = np.dot(vec_current, vec_future) / (norm_current * norm_future)
            bearing_angle[i] = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    df['azimuth_body_arena'] = bearing_angle
    print('azimuth_body_arena column added to dataframe')
    return df


def get_distance_to_cricket(df, cm_per_pixel=0.23):
    """Calculates the distance from mouse midpoint to cricket."""
    df['cricket_distance'] = lineardistance(df['cricket_x'], df['mid_x'], df['cricket_y'], df['mid_y']) * cm_per_pixel
    print('cricket_distance column added to dataframe')
    return df


def get_mouse_speed(df, frame_rate=30, cm_per_pixel=0.23, smooth_frames=15, smooth_order=3):
    """Calculates the speed of the mouse based on tailbase movement."""
    delta_mid_x = np.diff(df['tailbase_x'].values)
    delta_mid_y = np.diff(df['tailbase_y'].values)

    position_change = np.sqrt(delta_mid_x**2 + delta_mid_y**2) * cm_per_pixel

    speed = np.zeros(len(df))
    speed[1:] = position_change * frame_rate

    speed = savgol_filter(speed, smooth_frames, smooth_order)
    speed[speed < 0] = 0

    df['mouse_speed'] = speed
    print('mouse_speed column added to dataframe')
    return df


def get_mouse_acceleration(df, smooth_frames=15, smooth_order=3):
    """Calculates the acceleration of the mouse."""
    acceleration = np.gradient(df['mouse_speed'].values)
    acceleration = savgol_filter(acceleration, smooth_frames, smooth_order)
    df['mouse_acceleration'] = acceleration
    return df


def get_cricket_speed(df, frame_rate=30, cm_per_pixel=0.23, smooth_frames=15, smooth_order=3):
    """Calculates the speed of the cricket."""
    delta_cricket_x = np.diff(df['cricket_x'].values)
    delta_cricket_y = np.diff(df['cricket_y'].values)

    position_change = np.sqrt(delta_cricket_x**2 + delta_cricket_y**2) * cm_per_pixel

    speed = np.zeros(len(df))
    speed[1:] = position_change * frame_rate

    speed = savgol_filter(speed, smooth_frames, smooth_order)
    speed[speed < 0] = 0

    df['cricket_speed'] = speed
    print('cricket_speed column added to dataframe')
    return df

# ------------------------
# Events & epochs
# ------------------------

def get_contacts(df, contact_distance=4):
    """Identifies frames where mouse is in contact with the cricket."""
    df['contact'] = (df['cricket_distance'] < contact_distance).astype(int)
    print('contact column added to dataframe')
    return df


def smooth_contacts(df, windowSize=8):
    """Smooths the contact binary signal using a moving average."""
    if 'contact' not in df.columns or df['contact'].sum() == 0:
        print('No contacts to smooth.')
        return df

    weights = np.repeat(1.0, windowSize) / windowSize
    yMA = np.convolve(df['contact'].values, weights, 'same')

    df['contact'] = (yMA >= 1 / windowSize).astype(int)
    print('contact column smoothed')
    return df


def get_approaches(df, speed_threshold=10, diff_frames=4, diff_speed=-20, frame_rate=30, body_azimuth=60):
    """Identifies approach epochs based on speed, distance change, and body azimuth."""
    dist_diff = df['cricket_distance'].diff().rolling(diff_frames).mean()

    is_fast_enough = df['mouse_speed'].rolling(diff_frames).mean() > speed_threshold
    is_oriented = abs(df['azimuth_body']).rolling(diff_frames).mean() < body_azimuth
    is_closing_dist = dist_diff < (diff_speed / frame_rate)  # scaled by fps

    df['approach'] = (is_fast_enough & is_oriented & is_closing_dist).astype(int)
    return df


def smooth_approaches(df, windowSize=8):
    """Smooths the approach binary signal using a moving average."""
    if 'approach' not in df.columns or df['approach'].sum() == 0:
        print('No approaches to smooth.')
        return df

    weights = np.repeat(1.0, windowSize) / windowSize
    yMA = np.convolve(df['approach'].values, weights, 'same')

    df['approach'] = (yMA >= 1 / windowSize).astype(int)
    print('approach column smoothed')
    return df


def get_event_start_indices(df, event_col, to_capture_time=True, from_start_time=True):
    """
    Get absolute start indices for an event (e.g., 'contact', 'approach') within the analysis window.
    """
    series = _slice_by_window(df, from_start_time, to_capture_time)[event_col]
    starts_mask = series.diff().eq(1)
    # Include events already active at the first frame of the selected window.
    if not series.empty and series.iloc[0] == 1:
        starts_mask.iloc[0] = True
    return series.index[starts_mask].to_numpy()


def get_event_end_indices(df, event_col, to_capture_time=True, from_start_time=True):
    """Get absolute end indices for an event within the analysis window."""
    series = _slice_by_window(df, from_start_time, to_capture_time)[event_col]
    ends_mask = series.diff().eq(-1)
    ends = series.index[ends_mask].to_numpy()

    starts = get_event_start_indices(df, event_col, to_capture_time=to_capture_time, from_start_time=from_start_time)

    if len(ends) < len(starts):
        cap_idx = get_capture_frame(df) if to_capture_time else (series.index[-1] + 1)
        ends = np.append(ends, cap_idx)

    return ends


def get_contact_start_indices(df, to_capture_time=True, from_start_time=True):
    return get_event_start_indices(df, 'contact', to_capture_time, from_start_time)


def get_contact_end_indices(df, to_capture_time=True, from_start_time=True):
    return get_event_end_indices(df, 'contact', to_capture_time, from_start_time)


def get_approach_start_indices(df, to_capture_time=True, from_start_time=True):
    return get_event_start_indices(df, 'approach', to_capture_time, from_start_time)


def get_approach_end_indices(df, to_capture_time=True, from_start_time=True):
    return get_event_end_indices(df, 'approach', to_capture_time, from_start_time)


def get_capture_time(df):
    """Returns the time of the capture event (NaN if none)."""
    capture_frame = get_capture_frame(df)
    if capture_frame < len(df):
        return df.iloc[capture_frame]['time']
    return np.nan


def get_first_approach_time(df, to_capture_time=True, from_start_time=True):
    """Gets time of first approach within the analysis window."""
    try:
        win = _slice_by_window(df, from_start_time, to_capture_time)
        first_idx = win['approach'].idxmax()
        if win.loc[first_idx, 'approach'] == 1:
            return df.loc[first_idx, 'time']
    except Exception:
        pass
    return np.nan


def get_first_contact_time(df, to_capture_time=True, from_start_time=True):
    """Gets time of first contact within the analysis window."""
    try:
        win = _slice_by_window(df, from_start_time, to_capture_time)
        first_idx = win['contact'].idxmax()
        if win.loc[first_idx, 'contact'] == 1:
            return df.loc[first_idx, 'time']
    except Exception:
        pass
    return np.nan

# ------------------------
# Summaries & metrics
# ------------------------

def get_number_of_contacts(df, to_capture_time=True, from_start_time=True):
    return len(get_contact_start_indices(df, to_capture_time, from_start_time))


def get_number_of_approaches(df, to_capture_time=True, from_start_time=True):
    return len(get_approach_start_indices(df, to_capture_time, from_start_time))


def get_approach_intervals(df, to_capture_time=True, from_start_time=True, fps=30):
    """Intervals between approaches (seconds) within the window."""
    approach_starts = get_approach_start_indices(df, to_capture_time, from_start_time)
    approach_ends = get_approach_end_indices(df, to_capture_time, from_start_time)
    contact_ends = get_contact_end_indices(df, to_capture_time, from_start_time)

    if len(approach_starts) <= 1:
        return np.array([np.nan, np.nan])

    intervals = []
    for i in range(1, len(approach_starts)):
        current_start = approach_starts[i]
        last_approach_end = approach_ends[i - 1]

        previous_contact_ends = contact_ends[contact_ends < current_start]
        last_event_end = last_approach_end
        if previous_contact_ends.size > 0:
            last_contact_end = previous_contact_ends[-1]
            last_event_end = max(last_approach_end, last_contact_end)

        intervals.append(current_start - last_event_end)

    if not intervals:
        return np.array([np.nan, np.nan])

    intervals_in_sec = np.array(intervals) / fps
    return np.array([np.median(intervals_in_sec), np.mean(intervals_in_sec)])


def get_approach_path_to_border(df, to_capture_time=True, from_start_time=True):
    """Median/mean path-to-border sampled at approach starts within the window."""
    approach_starts = get_approach_start_indices(df, to_capture_time, from_start_time)
    if approach_starts.size == 0:
        return np.nan, np.nan
    path_distances = df.loc[approach_starts, 'path_to_borders']
    return np.median(path_distances), np.mean(path_distances)


def get_time_in_contact(df, to_capture_time=True, from_start_time=True):
    """Total time spent in contact within the window."""
    dfw = _slice_by_window(df, from_start_time, to_capture_time)
    frame_interval = dfw['time'].diff().mean()
    if pd.isna(frame_interval):
        frame_interval = 1 / 30.0
    return dfw['contact'].sum() * frame_interval


def get_time_exploring(df, to_capture_time=True, from_start_time=True):
    """Total time exploring (not approach/contact) within the window."""
    dfw = _slice_by_window(df, from_start_time, to_capture_time)

    frame_interval = dfw['time'].diff().mean()
    if pd.isna(frame_interval):
        frame_interval = 1 / 30.0

    exploring_frames_count = dfw[(dfw['approach'] == 0) & (dfw['contact'] == 0)].shape[0]
    return exploring_frames_count * frame_interval


def get_distribution(df, dist_bins, to_capture_time=True, from_start_time=True):
    """Distribution of mouse/cricket distances to border within the window."""
    dfw = _slice_by_window(df, from_start_time, to_capture_time)

    explore_mask = (dfw['approach'] == 0) & (dfw['contact'] == 0)
    approach_mask = dfw['approach'] == 1
    contact_mask = dfw['contact'] == 1
    approach_contact_mask = (dfw['approach'] == 1) | (dfw['contact'] == 1)

    def compute_hist(col, mask):
        data = dfw.loc[mask, col].dropna()
        if data.empty:
            return np.zeros(len(dist_bins) - 1)
        hist, _ = np.histogram(data, bins=dist_bins, density=True)
        return hist

    mouse = compute_hist('mouse_to_borders', explore_mask)
    cricket = compute_hist('cricket_to_borders', explore_mask)
    mouse_approach = compute_hist('mouse_to_borders', approach_mask)
    cricket_approach = compute_hist('cricket_to_borders', approach_mask)
    mouse_contact = compute_hist('mouse_to_borders', contact_mask)
    cricket_contact = compute_hist('cricket_to_borders', contact_mask)
    mouse_approach_contact = compute_hist('mouse_to_borders', approach_contact_mask)
    cricket_approach_contact = compute_hist('cricket_to_borders', approach_contact_mask)

    dist_dict = {
        'dist_to_border': dist_bins[:-1] + (np.diff(dist_bins) / 2),
        'mouse': mouse,
        'cricket': cricket,
        'mouse_approach': mouse_approach,
        'cricket_approach': cricket_approach,
        'mouse_contact': mouse_contact,
        'cricket_contact': cricket_contact,
        'mouse_approach_contact': mouse_approach_contact,
        'cricket_approach_contact': cricket_approach_contact,
    }

    return pd.DataFrame(dist_dict)


def get_density(df, bin_num, bin_range, to_capture_time=True, from_start_time=True):
    """2D spatial density histograms within the window."""
    dfw = _slice_by_window(df, from_start_time, to_capture_time)

    explore_mask = (dfw['approach'] == 0) & (dfw['contact'] == 0)
    approach_mask = dfw['approach'] == 1
    contact_mask = dfw['contact'] == 1
    approach_contact_mask = (dfw['approach'] == 1) | (dfw['contact'] == 1)

    def compute_hist(x_col, y_col, mask):
        x = dfw.loc[mask, x_col].dropna()
        y = dfw.loc[mask, y_col].dropna()
        if x.empty or y.empty:
            return np.zeros((bin_num, bin_num))
        hist, _, _ = np.histogram2d(x, y, bins=bin_num, range=bin_range, density=True)
        return hist

    mouse_explore = compute_hist('madj_x', 'madj_y', explore_mask)
    cricket_explore = compute_hist('cadj_x', 'cadj_y', explore_mask)
    mouse_approach = compute_hist('madj_x', 'madj_y', approach_mask)
    cricket_approach = compute_hist('cadj_x', 'cadj_y', approach_mask)
    mouse_contact = compute_hist('madj_x', 'madj_y', contact_mask)
    cricket_contact = compute_hist('cadj_x', 'cadj_y', contact_mask)
    mouse_approach_contact = compute_hist('madj_x', 'madj_y', approach_contact_mask)
    cricket_approach_contact = compute_hist('cadj_x', 'cadj_y', approach_contact_mask)

    dens_dict = {
        'mouse': mouse_explore.flatten(),
        'cricket': cricket_explore.flatten(),
        'mouse_approach': mouse_approach.flatten(),
        'cricket_approach': cricket_approach.flatten(),
        'mouse_contact': mouse_contact.flatten(),
        'cricket_contact': cricket_contact.flatten(),
        'mouse_approach_contact': mouse_approach_contact.flatten(),
        'cricket_approach_contact': cricket_approach_contact.flatten(),
    }

    return pd.DataFrame(dens_dict)


def get_azimuth_hist(df, azimuth_bins, max_dist_azimuth, to_capture_time=True, from_start_time=True):
    """Histograms of head/body azimuth for states within the window."""
    dfw = _slice_by_window(df, from_start_time, to_capture_time)

    explore_mask = (dfw['approach'] == 0) & (dfw['contact'] == 0)
    approach_mask = dfw['approach'] == 1
    contact_mask = dfw['contact'] == 1
    approach_contact_mask = (dfw['approach'] == 1) | (dfw['contact'] == 1)
    max_dist_mask = (approach_contact_mask) & (dfw['cricket_distance'] <= max_dist_azimuth)

    def compute_hist(col, mask):
        data = dfw.loc[mask, col].dropna()
        if data.empty:
            return np.zeros(len(azimuth_bins) - 1)
        hist, _ = np.histogram(data, bins=azimuth_bins, density=True)
        return hist

    azimuth_head_explore = compute_hist('azimuth_head', explore_mask)
    azimuth_body_explore = compute_hist('azimuth_body', explore_mask)
    azimuth_head_approach = compute_hist('azimuth_head', approach_mask)
    azimuth_body_approach = compute_hist('azimuth_body', approach_mask)
    azimuth_head_contact = compute_hist('azimuth_head', contact_mask)
    azimuth_body_contact = compute_hist('azimuth_body', contact_mask)
    azimuth_head_approach_contact = compute_hist('azimuth_head', approach_contact_mask)
    azimuth_body_approach_contact = compute_hist('azimuth_body', approach_contact_mask)
    azimuth_head_max_dist = compute_hist('azimuth_head', max_dist_mask)
    azimuth_body_max_dist = compute_hist('azimuth_body', max_dist_mask)

    azimuth_dict = {
        'azimuth_angle': azimuth_bins[:-1] + (np.diff(azimuth_bins) / 2),
        'azimuth_head_explore': azimuth_head_explore,
        'azimuth_body_explore': azimuth_body_explore,
        'azimuth_head_approach': azimuth_head_approach,
        'azimuth_body_approach': azimuth_body_approach,
        'azimuth_head_contact': azimuth_head_contact,
        'azimuth_body_contact': azimuth_body_contact,
        'azimuth_head_approach_contact': azimuth_head_approach_contact,
        'azimuth_body_approach_contact': azimuth_body_approach_contact,
        'azimuth_head_max_dist': azimuth_head_max_dist,
        'azimuth_body_max_dist': azimuth_body_max_dist
    }

    return pd.DataFrame(azimuth_dict)


def get_capture_time_relative_to_first_approach_contact(df):
    """Capture time relative to the first approach or contact (window-aware via defaults)."""
    first_event_time = np.nanmin([get_first_contact_time(df), get_first_approach_time(df)])
    if np.isnan(first_event_time):
        return np.nan
    return get_capture_time(df) - first_event_time


def add_pre_contact(df, pre_frames=15):
    """Adds a 'pre_contact' column for frames immediately preceding a contact."""
    df['pre_contact'] = 0
    contact_starts = get_contact_start_indices(df, to_capture_time=False, from_start_time=False)
    for start in contact_starts:
        pre_start = max(0, start - pre_frames)
        df.loc[pre_start:start, 'pre_contact'] = 1
    return df


def get_p_contact_given_approach(df, pre_frames=15):
    """Probability that a contact occurs shortly after an approach ends (window-aware approach ends)."""
    if 'pre_contact' not in df.columns:
        df = add_pre_contact(df, pre_frames)

    approach_ends = get_approach_end_indices(df, to_capture_time=True, from_start_time=True)
    if not approach_ends.size:
        return 0.0

    approach_contact = 0
    for end_idx in approach_ends:
        if end_idx < len(df) and df.loc[end_idx, 'pre_contact'] == 1:
            approach_contact += 1

    return approach_contact / len(approach_ends)


def summarize_df(df, trial_id="", condition=""):
    """Creates a one-row summary DataFrame for the trial (window-aware)."""
    num_approaches = get_number_of_approaches(df, to_capture_time=True, from_start_time=True)
    num_contacts = get_number_of_contacts(df, to_capture_time=True, from_start_time=True)
    approach_intervals = get_approach_intervals(df, to_capture_time=True, from_start_time=True)
    median_border, mean_border = get_approach_path_to_border(df, to_capture_time=True, from_start_time=True)

    capture_time = get_capture_time(df)
    total_exploration_time = get_time_exploring(df, to_capture_time=True, from_start_time=True)

    summary_data = {
        'trial_id': trial_id,
        'condition': condition,
        'capture_time': capture_time,
        'capture_time_first_approach_contact': get_capture_time_relative_to_first_approach_contact(df),
        'number_of_approaches': num_approaches,
        'p_contact_approach': get_p_contact_given_approach(df, pre_frames=15),
        'p_capture_approach': 1 / num_approaches if num_approaches > 0 else 0,
        'first_approach_latency': get_first_approach_time(df, to_capture_time=True, from_start_time=True),
        'median_approach_interval': approach_intervals[0],
        'mean_approach_interval': approach_intervals[1],
        'median_approach_to_border': median_border,
        'mean_approach_to_border': mean_border,
        'number_of_contacts': num_contacts,
        'p_capture_contact': 1 / num_contacts if num_contacts > 0 else 0,
        'time_in_contact': get_time_in_contact(df, to_capture_time=True, from_start_time=True),
        'first_contact_latency': get_first_contact_time(df, to_capture_time=True, from_start_time=True),
        'total_time_exploring': total_exploration_time,
        'fraction_time_exploring': total_exploration_time / capture_time if capture_time and capture_time > 0 else 0,
    }

    summary_df = pd.DataFrame([summary_data])
    return summary_df

# ------------------------
# Save path helpers
# ------------------------

def get_save_path_csv(h5_file):
    """Generates a full save path for the main analysis CSV in the same directory."""
    dir_path = os.path.dirname(h5_file)
    base_name = os.path.splitext(os.path.basename(h5_file))[0]
    return os.path.join(dir_path, f"{base_name}_analysis_full.csv")


def get_save_path_csv_summary(h5_file):
    """Generates a full save path for the summary CSV."""
    dir_path = os.path.dirname(h5_file)
    base_name = os.path.splitext(os.path.basename(h5_file))[0]
    return os.path.join(dir_path, f"{base_name}_summary.csv")


def get_save_path_csv_distribution(h5_file):
    """Generates a full save path for the distribution CSV."""
    dir_path = os.path.dirname(h5_file)
    base_name = os.path.splitext(os.path.basename(h5_file))[0]
    return os.path.join(dir_path, f"{base_name}_distribution.csv")


def get_save_path_csv_density(h5_file):
    """Generates a full save path for the density CSV."""
    dir_path = os.path.dirname(h5_file)
    base_name = os.path.splitext(os.path.basename(h5_file))[0]
    return os.path.join(dir_path, f"{base_name}_density.csv")


def get_save_path_csv_azimuth(h5_file):
    """Generates a full save path for the azimuth CSV."""
    dir_path = os.path.dirname(h5_file)
    base_name = os.path.splitext(os.path.basename(h5_file))[0]
    return os.path.join(dir_path, f"{base_name}_azimuth.csv")

# ------------------------
# Plotting & video (window-aware)
# ------------------------

def plot_hunt(df, to_capture_time=True, from_start_time=True, video_path="", save_fig=False):
    """Plots spatial paths of mouse and cricket within the window."""
    df_to_plot = _slice_by_window(df, from_start_time, to_capture_time)

    fig, ax = plt.subplots()
    ax.plot(df_to_plot['madj_x'], df_to_plot['madj_y'], c='dodgerblue', zorder=2, label="mouse")
    ax.plot(df_to_plot['cadj_x'], df_to_plot['cadj_y'], c='k', zorder=1, label="cricket")
    ax.scatter(df_to_plot['madj_x'].iloc[0], df_to_plot['madj_y'].iloc[0], c='cyan', marker='^', zorder=3, label="mouse_start")
    ax.scatter(df_to_plot['madj_x'].iloc[-1], df_to_plot['madj_y'].iloc[-1], c='cyan', marker='p', zorder=3, label="mouse_end")
    ax.scatter(df_to_plot['cadj_x'].iloc[0], df_to_plot['cadj_y'].iloc[0], c='red', marker='^', zorder=3, label="cricket_start")
    ax.scatter(df_to_plot['cadj_x'].iloc[-1], df_to_plot['cadj_y'].iloc[-1], c='red', marker='p', zorder=3, label="cricket_end")

    plot_title = os.path.basename(video_path).split('.')[0] + '_hunt' if video_path else 'hunt'
    ax.set_title(plot_title)
    ax.legend(loc="best")
    ax.set_aspect('equal', adjustable='box')

    if save_fig:
        fig.savefig(plot_title + '.png', dpi=300)

    plt.close(fig)
    return


def plot_approaches(df, to_capture_time=True, from_start_time=True, video_path="", save_fig=False):
    """Plots approach paths of the mouse within the window."""
    starts = get_approach_start_indices(df, to_capture_time, from_start_time)
    ends = get_approach_end_indices(df, to_capture_time, from_start_time)

    fig, ax = plt.subplots()
    if starts.size > 0:
        for start, end in zip(starts, ends):
            ax.plot(df.loc[start:end, 'madj_x'], df.loc[start:end, 'madj_y'], c='dodgerblue', zorder=2)

    plot_title = os.path.basename(video_path).split('.')[0] + '_approaches' if video_path else 'approaches'
    ax.set_title(plot_title)
    ax.set_aspect('equal', adjustable='box')

    if save_fig:
        fig.savefig(plot_title + '.png', dpi=300)

    plt.close(fig)
    return


def plot_speeds_and_distance(df, mouse=True, cricket=True, contact_distance=4, to_capture_time=False, from_start_time=True, show_approaches=True, show_contacts=True, video_path="", save_fig=False):
    """Plots speeds and distance to cricket over time within the window."""
    df_to_plot = _slice_by_window(df, from_start_time, to_capture_time).copy()

    df_to_plot.loc[df_to_plot['mouse_speed'] < 0, 'mouse_speed'] = 0
    df_to_plot.loc[df_to_plot['cricket_speed'] < 0, 'cricket_speed'] = 0
    df_to_plot.loc[df_to_plot['cricket_distance'] < 0, 'cricket_distance'] = 0

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    time_axis = df_to_plot['time']

    if mouse:
        ax1.plot(time_axis, df_to_plot['mouse_speed'], label='mouse speed', color='dodgerblue')
    if cricket:
        ax1.plot(time_axis, df_to_plot['cricket_speed'], label='cricket speed', color='orange')

    ax2.plot(time_axis, df_to_plot['cricket_distance'], label="distance to cricket", color="green")
    ax2.axhline(y=contact_distance, linestyle='--', color='r', label='contact threshold')

    max_speed_val = df_to_plot[['mouse_speed', 'cricket_speed']].max().max()
    if np.isnan(max_speed_val) or max_speed_val == 0:
        max_speed_val = 1

    if show_approaches:
        ax1.fill_between(time_axis, 0, max_speed_val, where=df_to_plot['approach'] > 0, facecolor='green', alpha=0.3, label='approach')
    if show_contacts:
        ax1.fill_between(time_axis, 0, max_speed_val, where=df_to_plot['contact'] > 0, facecolor='red', alpha=0.3, label='contact')

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Speed (cm/s)')
    ax2.set_ylabel('Distance (cm)')
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    ax1.set_ylim(0, max_speed_val * 1.1)
    ax2.set_ylim(0)
    ax1.set_xlim(0, time_axis.iloc[-1] if not time_axis.empty else 1)

    plot_title = os.path.basename(video_path).split('.')[0] + '_speed_and_distance' if video_path else 'speed_and_distance'
    plt.title(plot_title)

    if save_fig:
        fig.savefig(plot_title + '.png', dpi=300)

    plt.close(fig)
    return


def plot_azimuth_hist(df, n_bins=20, approach_only=True, to_capture_time=True, from_start_time=True, video_path="", save_fig=False):
    """Plots a histogram of the head azimuth within the window."""
    dfw = _slice_by_window(df, from_start_time, to_capture_time)
    azimuth = dfw.loc[dfw['approach'] == 1, 'azimuth_head'] if approach_only else dfw['azimuth_head']

    fig, ax = plt.subplots()
    ax.hist(azimuth.dropna(), n_bins, density=True)
    ax.set_xlabel('Azimuth (deg)')
    ax.set_ylabel('Probability Density')
    ax.set_xlim(-180, 180)
    ax.axvline(x=25, color='orange', linestyle='--')
    ax.axvline(x=-25, color='orange', linestyle='--')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    approach_str = 'approach_only' if approach_only else 'all_frames'
    plot_title = f"{os.path.basename(video_path).split('.')[0]}_azimuth_{approach_str}" if video_path else f"azimuth_{approach_str}"
    plt.title(plot_title)

    if save_fig:
        fig.savefig(plot_title + '.png', dpi=300)

    plt.close(fig)
    return

# ------------------------
# Arena setup & transforms
# ------------------------

def select_arena_manual(video_path, frame_number=0):
    """Displays a frame from a video and allows the user to select arena corners."""
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    arena_image_dir = 'arena_images'
    if not os.path.exists(arena_image_dir):
        os.makedirs(arena_image_dir)

    video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = video.read()
    video.release()

    if not ret:
        raise IOError(f"Could not read frame {frame_number} from video.")

    image_path = os.path.join(arena_image_dir, f'frame{frame_number}.jpg')
    cv2.imwrite(image_path, frame)
    print(f'Creating... {image_path}')

    img = plt.imread(image_path)
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title("Select TL, TR, BL, BR corners. Close window to finish.", color='red')

    corners = plt.ginput(4, timeout=-1)
    plt.close(fig)

    print("Corners selected.")
    return corners


def pixel_size_from_arena_coordinates(arena_coordinates, arena_size_x=45, arena_size_y=38):
    """Calculates the cm/pixel conversion factor from arena corner coordinates."""
    if len(arena_coordinates) < 4:
        print("Error: Not enough corner points provided.")
        return 0.1

    tl, tr, bl, br = arena_coordinates

    pixel_width = ((tr[0] - tl[0]) + (br[0] - bl[0])) / 2
    pixel_height = ((bl[1] - tl[1]) + (br[1] - tr[1])) / 2

    px_per_cm_x = pixel_width / arena_size_x if arena_size_x else 1
    px_per_cm_y = pixel_height / arena_size_y if arena_size_y else 1

    cm_per_px_x = 1 / px_per_cm_x if px_per_cm_x else 0
    cm_per_px_y = 1 / px_per_cm_y if px_per_cm_y else 0

    print(f"cm per pixel (x-axis): {cm_per_px_x:.4f}")
    print(f"cm per pixel (y-axis): {cm_per_px_y:.4f}")

    pixel_size = np.mean([cm_per_px_x, cm_per_px_y])
    return pixel_size


def affine_transform(video_corners, target_corners, video_points):
    """Applies a perspective transform to video points."""
    trans_matrix = cv2.getPerspectiveTransform(np.float32(video_corners), np.float32(target_corners))

    num_points = video_points.shape[1]
    video_points_3d = np.vstack([video_points, np.ones(num_points)])

    target_points_3d = np.matmul(trans_matrix, video_points_3d)

    w = target_points_3d[2, :]
    w[w == 0] = 1e-6
    target_points = target_points_3d[:2, :] / w

    return target_points.T


def add_corners(df, corners):
    """Adds the four arena corner coordinates to the first four rows of the DataFrame."""
    corners_x = pd.Series(np.nan, index=df.index)
    corners_y = pd.Series(np.nan, index=df.index)

    for i in range(min(4, len(corners))):
        corners_x.iloc[i] = corners[i][0]
        corners_y.iloc[i] = corners[i][1]

    df['corners_x'] = corners_x
    df['corners_y'] = corners_y
    return df


def get_borders(corners, pts_per_border=1000):
    """Generates an array of points outlining the borders of the arena."""
    new_order = [corners[0], corners[1], corners[3], corners[2], corners[0]]
    borders = np.zeros((4 * pts_per_border, 2))

    for i in range(4):
        start_pt = new_order[i]
        end_pt = new_order[i + 1]
        borders[i * pts_per_border:(i + 1) * pts_per_border, 0] = np.linspace(start_pt[0], end_pt[0], pts_per_border)
        borders[i * pts_per_border:(i + 1) * pts_per_border, 1] = np.linspace(start_pt[1], end_pt[1], pts_per_border)

    return borders

# ------------------------
# Video annotation (window-aware)
# ------------------------

def annotate_video(df, video_path, fps_out=30, to_capture_time=True, from_start_time=True, show_time=True, show_borders=True, label_bodyparts=True, show_approaches=True, show_approach_number=True, show_contacts=True, show_contact_number=True, show_azimuth=True, show_azimuth_lines=True, show_distance=True, show_speed=True, save_video=True, show_video=True, border_size=40, video_path_ext='', transform_to_arena=False, arena_width_cm=45, arena_height_cm=38):
    """Annotates the video with analysis data, restricted to the analysis window.

    If transform_to_arena=True and corner coordinates are present in df['corners_x/y'],
    each frame is perspective-warped to an arena-aligned view before writing.
    """
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_width = frame_width + 2 * border_size
    out_height = frame_height + 2 * border_size

    transform_matrix = None
    target_rect = None
    if transform_to_arena:
        corner_df = df[['corners_x', 'corners_y']].dropna().head(4) if {'corners_x', 'corners_y'}.issubset(df.columns) else pd.DataFrame()
        if len(corner_df) == 4:
            video_corners = np.float32(corner_df.values)

            arena_aspect_ratio = float(arena_width_cm) / float(arena_height_cm) if arena_height_cm else 1.0
            drawable_width = max(1, out_width - 2 * border_size)
            drawable_height = max(1, out_height - 2 * border_size)
            drawable_aspect_ratio = drawable_width / drawable_height

            if drawable_aspect_ratio > arena_aspect_ratio:
                target_h = drawable_height
                target_w = target_h * arena_aspect_ratio
                padding_x = (out_width - target_w) / 2.0
                padding_y = float(border_size)
            else:
                target_w = drawable_width
                target_h = target_w / arena_aspect_ratio if arena_aspect_ratio else drawable_height
                padding_x = float(border_size)
                padding_y = (out_height - target_h) / 2.0

            target_corners = np.float32([
                [padding_x, padding_y],
                [padding_x + target_w, padding_y],
                [padding_x, padding_y + target_h],
                [padding_x + target_w, padding_y + target_h]
            ])
            transform_matrix = cv2.getPerspectiveTransform(video_corners, target_corners)
            target_rect = (
                (int(round(padding_x)), int(round(padding_y))),
                (int(round(padding_x + target_w)), int(round(padding_y + target_h)))
            )
        else:
            print("Warning: transform_to_arena=True but 4 valid corners were not found; using unwarped annotation.")
            transform_to_arena = False

    video_out = None
    if save_video:
        save_path = os.path.splitext(video_path)[0] + video_path_ext + ".avi"
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        video_out = cv2.VideoWriter(save_path, fourcc, fps_out, (out_width, out_height))
        print(f"Saving annotated video to: {save_path}")

    # Set video to start of analysis window
    start_idx = get_start_frame(df) if from_start_time else 0
    if start_idx > 0:
        video.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

    frame_idx = start_idx
    while video.isOpened():
        ret, frame = video.read()
        if not ret or frame_idx >= len(df):
            break

        # Stop at capture frame if specified
        if to_capture_time and df.loc[frame_idx, 'captured'] == 1:
            break

        # Overlay points/lines
        if label_bodyparts:
            nose_pos = (int(df.loc[frame_idx, 'nose_x']), int(df.loc[frame_idx, 'nose_y'])) if pd.notnull(df.loc[frame_idx, 'nose_x']) else None
            cricket_pos = (int(df.loc[frame_idx, 'cricket_x']), int(df.loc[frame_idx, 'cricket_y'])) if pd.notnull(df.loc[frame_idx, 'cricket_x']) else None

            if nose_pos:
                cv2.circle(frame, nose_pos, 3, (255, 0, 100), -1)
            if cricket_pos:
                cv2.circle(frame, cricket_pos, 3, (100, 255, 0), -1)

        if show_azimuth_lines:
            mid_pos = (int(df.loc[frame_idx, 'mid_x']), int(df.loc[frame_idx, 'mid_y'])) if pd.notnull(df.loc[frame_idx, 'mid_x']) else None
            cricket_pos = (int(df.loc[frame_idx, 'cricket_x']), int(df.loc[frame_idx, 'cricket_y'])) if pd.notnull(df.loc[frame_idx, 'cricket_x']) else None
            if mid_pos and cricket_pos:
                cv2.line(frame, mid_pos, cricket_pos, (0, 255, 0), 1)

        if transform_to_arena and transform_matrix is not None:
            bordered_frame = cv2.warpPerspective(frame, transform_matrix, (out_width, out_height))
            if show_borders and target_rect is not None:
                cv2.rectangle(bordered_frame, target_rect[0], target_rect[1], (255, 255, 255), 2)
        else:
            bordered_frame = cv2.copyMakeBorder(frame, top=border_size, bottom=border_size, left=border_size, right=border_size, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])

        # Border text
        if show_time:
            cv2.putText(bordered_frame, f"Time: {df.loc[frame_idx, 'time']:.2f}s", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if show_approaches and df.loc[frame_idx, 'approach']:
            cv2.putText(bordered_frame, "APPROACH", (150, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if show_contacts and df.loc[frame_idx, 'contact']:
            cv2.putText(bordered_frame, "CONTACT", (280, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if show_video:
            cv2.imshow('Annotated Video', bordered_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if video_out:
            video_out.write(bordered_frame)

        frame_idx += 1

    video.release()
    if video_out:
        video_out.release()
    cv2.destroyAllWindows()
    print("Annotation finished.")
    return
