# -*- coding: utf-8 -*-
"""
@author: @David Kerschensteiner and @Daniel Kerschensteiner
Manual Data Correction and Video Inspection Tool

This script provides a GUI to manually correct DeepLabCut tracking data.
This version includes a comprehensive set of key codes to ensure arrow key
navigation works reliably across different operating systems (macOS, Windows, Linux).

Key Features:
- Smartly loads either original DLC csv files or previously saved _pythonAnalysis.csv files.
- Robust and reliable keyboard shortcuts for frame navigation.
- Simple frame caching to prevent redundant video reads and improve performance.
- Interactive GUI for selecting body parts and correcting labels.
- Produces a final CSV file that precisely matches the specified output format.
- Applies median filtering to specific body parts if not previously applied.
"""

import os
import sys
import cv2
import pandas as pd
import numpy as np
import scipy.ndimage as ndimage
from tkinter import filedialog, Tk, messagebox
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QRadioButton,
                             QVBoxLayout, QPushButton, QButtonGroup, QSlider, QHBoxLayout)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QWheelEvent, QMouseEvent


def load_and_prepare_dataframe(csv_path):
    """
    Loads a DeepLabCut CSV file or a previously saved analysis file.
    """
    try:
        if csv_path.endswith('_pythonAnalysis.csv'):
            print("Loading previously corrected analysis file...")
            df = pd.read_csv(csv_path, index_col='frame_number')
            print("File loaded successfully.")
        else:
            print("Loading original DeepLabCut file...")
            df = pd.read_csv(csv_path, header=[1, 2], index_col=0)
            new_columns = []
            for bodypart, coord in df.columns:
                if bodypart == 'l_ear':
                    bodypart = 'leftear'
                if bodypart == 'r_ear':
                    bodypart = 'rightear'
                if bodypart == 'tail_base':
                    bodypart = 'tailbase'
                new_columns.append(f"{bodypart}_{coord}")
            df.columns = new_columns
            df.index.name = 'frame_number'
            print("DataFrame loaded and formatted successfully.")

        # Apply median filtering if not previously applied, for specific body parts
        bodyparts_to_filter = ['leftear', 'rightear', 'nose', 'tailbase']
        for bp in bodyparts_to_filter:
            flag_col = f'median_filtered_{bp}'
            if flag_col not in df.columns:
                print(f"Applying median filter to {bp}")
                for coord in ['x', 'y']:
                    col = f'{bp}_{coord}'
                    if col in df.columns:
                        df[col] = ndimage.median_filter(
                            df[col].values, size=3, mode='nearest')
                df[flag_col] = True  # Add flag column
            else:
                print(f"{bp} already median filtered, skipping.")

        return df
    except Exception as e:
        print(f"Error loading or formatting CSV file: {e}")
        messagebox.showerror(
            "File Error", f"Could not process the CSV file: {e}\n\nPlease ensure it is a valid CSV file.")
        return None


class CorrectionController(QWidget):
    """A PyQt5 window to control which bodypart is being labeled and display the video."""

    def __init__(self, bodyparts, df, video_path, save_path):
        super().__init__()
        self.bodyparts = bodyparts
        self.df = df
        self.frame_index = pd.Index(self.df.index)
        self.video_path = video_path
        self.save_path = save_path
        self.video = cv2.VideoCapture(video_path)
        if not self.video.isOpened():
            messagebox.showerror(
                "Video Error", f"Could not open video file: {video_path}")
            self.video = None
            return
        self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, frame = self.video.read()
        if ret:
            self.frame_height, self.frame_width = frame.shape[:2]
            self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            self.frame_height, self.frame_width = 480, 640  # Default
        self.current_part_index = 0
        self.current_frame_cache = {'frame_num': -1, 'image': None}
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Correction Controls')
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # Top controls in horizontal layout
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(5)

        self.label = QLabel(
            f"Select part to label.\nNow correcting: {self.bodyparts[self.current_part_index]}")
        controls_layout.addWidget(self.label)

        # Radio buttons in horizontal layout
        radios_layout = QHBoxLayout()
        self.radio_group = QButtonGroup()
        for i, part in enumerate(self.bodyparts):
            radio_button = QRadioButton(part)
            radios_layout.addWidget(radio_button)
            self.radio_group.addButton(radio_button, i)
            if i == 0:
                radio_button.setChecked(True)
        controls_layout.addLayout(radios_layout)

        # Buttons in horizontal layout
        buttons_layout = QHBoxLayout()
        self.next_button = QPushButton("Next Bodypart")
        self.next_button.clicked.connect(self.next_part)
        buttons_layout.addWidget(self.next_button)

        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save)
        buttons_layout.addWidget(self.save_button)
        controls_layout.addLayout(buttons_layout)

        main_layout.addLayout(controls_layout)

        # Slider above the video
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setRange(0, self.total_frames - 1)
        self.slider.valueChanged.connect(self.update_frame_display)
        main_layout.addWidget(self.slider)

        self.image_label = QLabel(self)
        self.image_label.setFixedSize(self.frame_width, self.frame_height)
        main_layout.addWidget(self.image_label)

        self.setLayout(main_layout)
        self.show()

        # Initial display
        self.update_frame_display(0)

    def get_current_selection(self):
        """Returns the column names for the currently selected bodypart."""
        selected_id = self.radio_group.checkedId()
        if selected_id < 0:
            selected_id = self.current_part_index
        part_name = self.bodyparts[selected_id]
        return f'{part_name}_x', f'{part_name}_y', f'{part_name}_likelihood'

    def next_part(self):
        """Cycles to the next bodypart in the list."""
        self.current_part_index = (
            self.current_part_index + 1) % len(self.bodyparts)
        self.radio_group.button(self.current_part_index).setChecked(True)
        self.label.setText(
            f"Select part to label.\nNow correcting: {self.bodyparts[self.current_part_index]}")
        print(
            f"Switched to correcting: {self.bodyparts[self.current_part_index]}")
        # Refresh to show new marker if applicable
        self.update_frame_display(self.slider.value())

    def save(self):
        """Saves the current corrections to the CSV file."""
        try:
            if 'time' not in self.df.columns:
                frame_rate = 30
                self.df['time'] = self.df.index / frame_rate

            final_column_order = [
                'time', 'leftear_x', 'leftear_y', 'leftear_likelihood',
                'rightear_x', 'rightear_y', 'rightear_likelihood',
                'nose_x', 'nose_y', 'nose_likelihood',
                'tailbase_x', 'tailbase_y', 'tailbase_likelihood',
                'cricket_x', 'cricket_y', 'cricket_likelihood'
            ]

            bodyparts_to_filter = ['leftear', 'rightear', 'nose', 'tailbase']
            flag_cols = [
                f'median_filtered_{bp}' for bp in bodyparts_to_filter if f'median_filtered_{bp}' in self.df.columns]

            output_df = self.df.reindex(columns=final_column_order + flag_cols)

            output_df.to_csv(self.save_path, index_label='frame_number')

            print(f"\nSuccessfully saved corrected data to:\n{self.save_path}")
            messagebox.showinfo(
                "Success", f"Corrected data saved to:\n{self.save_path}")

        except Exception as e:
            print(f"Error saving the CSV file: {e}")
            messagebox.showerror(
                "Save Error", f"Could not save the corrected data: {e}")

    def closeEvent(self, event):
        """Closes the application cleanly when the window is closed."""
        print("Finishing corrections.")
        if self.video:
            self.video.release()
        event.accept()
        QApplication.instance().quit()

    def update_frame_display(self, trackbar_value):
        """Reads a frame, draws labels, and displays it."""
        if self.video is None:
            return
        if trackbar_value != self.current_frame_cache['frame_num']:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, trackbar_value)
            ret, frame_img = self.video.read()
            if not ret:
                return
            self.current_frame_cache['frame_num'] = trackbar_value
            self.current_frame_cache['image'] = frame_img

        display_img = self.current_frame_cache['image'].copy()
        label_x_col, label_y_col, _ = self.get_current_selection()

        if trackbar_value in self.frame_index and pd.notna(self.df.loc[trackbar_value, label_x_col]):
            pos = (int(self.df.loc[trackbar_value, label_x_col]), int(
                self.df.loc[trackbar_value, label_y_col]))
            cv2.drawMarker(display_img, pos, color=(
                0, 0, 255), markerType=cv2.MARKER_CROSS, thickness=2, markerSize=15)

        cv2.putText(display_img, f"Frame: {trackbar_value}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_img, f"Correcting: {label_x_col.split('_')[0]}", (
            20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Convert to RGB for Qt
        display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        height, width, channels = display_img.shape
        bytes_per_line = channels * width
        q_img = QImage(display_img.data, width, height,
                       bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_img))

    def mousePressEvent(self, event: QMouseEvent):
        """Handles mouse clicks on the window, but check if on image."""
        if self.image_label.underMouse() and event.button() == Qt.LeftButton:
            # Get position relative to image_label
            pos = event.pos() - self.image_label.pos()
            x, y = pos.x(), pos.y()
            current_frame = self.slider.value()
            label_x_col, label_y_col, likelihood_col = self.get_current_selection()
            if current_frame not in self.frame_index:
                return
            print(
                f"Frame {current_frame}: Setting {label_x_col} to ({x}, {y})")
            self.df.loc[current_frame, label_x_col] = x
            self.df.loc[current_frame, label_y_col] = y
            self.df.loc[current_frame, likelihood_col] = 1.0
            self.update_frame_display(current_frame)

    def wheelEvent(self, event: QWheelEvent):
        """Handles mouse wheel for frame navigation."""
        delta = event.angleDelta().y()
        current_frame = self.slider.value()
        new_frame = current_frame - 1 if delta > 0 else current_frame + 1
        new_frame = np.clip(new_frame, 0, self.total_frames - 1)
        self.slider.setValue(new_frame)

    def keyPressEvent(self, event):
        """Handles key presses for frame navigation."""
        key = event.key()
        current_frame = self.slider.value()
        new_frame = None
        if key == Qt.Key_Left:
            new_frame = current_frame - 1
        elif key == Qt.Key_Right:
            new_frame = current_frame + 1
        elif key == Qt.Key_Q or key == Qt.Key_Escape:
            self.close()
            return
        if new_frame is not None:
            new_frame = np.clip(new_frame, 0, self.total_frames - 1)
            self.slider.setValue(new_frame)


def run_correction_interface(df, video_path, save_path):
    """
    Main interactive tool to correct label positions frame by frame.
    """
    app = QApplication.instance() or QApplication(sys.argv)

    bodyparts = sorted(
        {
            col[:-2]
            for col in df.columns
            if col.endswith('_x')
            and f"{col[:-2]}_y" in df.columns
            and f"{col[:-2]}_likelihood" in df.columns
        }
    )
    if not bodyparts:
        messagebox.showerror(
            "Data Error",
            "No valid bodypart columns found. Expected columns like nose_x/nose_y/nose_likelihood."
        )
        return df

    controller = CorrectionController(bodyparts, df, video_path, save_path)

    print("Starting manual correction...")
    print("- Use Left/Right arrow keys or the mouse wheel to navigate frames.")
    print("- Left-click on the video to set a new position for the selected body part.")
    print("- Use the 'Correction Controls' window to switch body parts or save.")
    print("- Close the window to exit.")

    app.exec_()
    print("Correction interface closed.")
    return df


def main():
    """Main function to run the entire correction and saving workflow."""
    root = Tk()
    root.withdraw()

    video_path = filedialog.askopenfilename(
        title='Choose the .mp4 video file', filetypes=[("MP4 files", "*.mp4")])
    if not video_path:
        print("No video file selected. Exiting.")
        return

    csv_path = filedialog.askopenfilename(
        title='Choose a data file (original DLC .csv or ..._pythonAnalysis.csv)',
        filetypes=[("CSV files", "*.csv")]
    )
    if not csv_path:
        print("No CSV file selected. Exiting.")
        return

    df = load_and_prepare_dataframe(csv_path)
    if df is None:
        return
    if 'frame_number' in df.columns:
        df = df.set_index('frame_number')
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    base_name_csv = os.path.splitext(os.path.basename(csv_path))[0]
    if base_name_csv.endswith('_pythonAnalysis'):
        base_name_csv = base_name_csv.replace('_pythonAnalysis', '')

    directory = os.path.dirname(csv_path)
    output_filename = f"{base_name_csv}_pythonAnalysis.csv"
    save_path = os.path.join(directory, output_filename)

    run_correction_interface(df, video_path, save_path)


if __name__ == '__main__':
    main()
