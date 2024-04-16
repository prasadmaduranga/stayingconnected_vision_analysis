import cv2
import mediapipe as mp
import argparse
import os
import json
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ipywidgets as widgets
from IPython.display import display

import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

sys.path.append('../')
from util import DatabaseUtil
from util.feature_util import *
from features.body_landmarks import BodyLandmark
from util  import calculate_angle,calculate_bbox,bbox_iou,calculate_distance,calculate_object_center,calculate_line_equation
import pandas as pd
import numpy as np



project_path = '/Users/prasadmaduranga/higher_studies/research/Stroke research/Projects/Staying connected Project/My Projects/stayingconnected_vision_analysis'

# HAND_BOUNDING_BOX_IOU
def plot_landmarks_for_frame(frame_seq_number):
    # Filter for the selected frame sequence number
    frame_df = df[df['frame_seq_number'] == frame_seq_number]

    # Initialize 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each landmark
    for landmark in landmarks:
        # Extract 3D coordinates
        x = frame_df[f'{landmark}_X'].values
        y = frame_df[f'{landmark}_Y'].values
        z = frame_df[f'{landmark}_Z'].values

        # Plot point
        ax.scatter(x, y, z, label=landmark,s=100)

        # Connect landmarks with lines if not the first landmark
        if landmarks.index(landmark) > 0:
            prev_landmark = landmarks[landmarks.index(landmark) - 1]
            prev_x = frame_df[f'{prev_landmark}_X'].values
            prev_y = frame_df[f'{prev_landmark}_Y'].values
            prev_z = frame_df[f'{prev_landmark}_Z'].values
            ax.plot([prev_x, x], [prev_y, y], [prev_z, z], 'r-')

    ax.legend()
    plt.show()



def fetch_frame_coordinates(ids):
    """
    Fetches frame coordinates for the given IDs from the database.
    """
    placeholders = ', '.join('?' for _ in ids)  # Create placeholders for parameterized query
    query = f"""
    SELECT id, recording_id, frame_rate, coordinates, coordinates_object1, coordinates_object2 
    FROM frame_coordinates
    WHERE id IN ({placeholders});
    """
    db_util = DatabaseUtil()
    return db_util.fetch_data(query, ids)

def transform_data_to_dataframe(data):
    expanded_data = []
    for record in data:
        frame_coordinate_id, recording_id, frame_rate, coordinates_json, coordinates_object1_json, coordinates_object2_json = record
        coordinates = json.loads(coordinates_json)
        coordinates_object1 = json.loads(coordinates_object1_json)
        coordinates_object2 = json.loads(coordinates_object2_json)

        for frame_seq_number, frame_coordinates in enumerate(coordinates):
            frame_data = {
                'frame_coordinate_id': frame_coordinate_id,
                'recording_id': recording_id,
                'frame_rate': frame_rate,
                'frame_seq_number': frame_seq_number,
            }
            for landmark in BodyLandmark:
                if landmark.value < len(frame_coordinates) and frame_coordinates[landmark.value] is not None:
                    frame_data[f'{landmark.name}_X'] = frame_coordinates[landmark.value].get('x')
                    frame_data[f'{landmark.name}_Y'] = frame_coordinates[landmark.value].get('y')
                    frame_data[f'{landmark.name}_Z'] = frame_coordinates[landmark.value].get('z')
                else:
                    frame_data[f'{landmark.name}_X'] = None
                    frame_data[f'{landmark.name}_Y'] = None
                    frame_data[f'{landmark.name}_Z'] = None

            # Process object 1 and 2 coordinates for the frame, if presents
            for obj_index, obj_coordinates in enumerate([coordinates_object1, coordinates_object2], start=1):
                if frame_seq_number < len(obj_coordinates):
                    obj_data = obj_coordinates[frame_seq_number]

                    try:
                        if obj_data[0] is None:
                            for coord in ['x1', 'y1', 'x2', 'y2']:
                                frame_data[f'object_{obj_index}_{coord}'] = None
                            continue

                        frame_data[f'object_{obj_index}_x1'] = obj_data[0]['x']
                        frame_data[f'object_{obj_index}_y1'] = obj_data[0]['y']
                        frame_data[f'object_{obj_index}_x2'] = obj_data[1]['x']
                        frame_data[f'object_{obj_index}_y2'] = obj_data[1]['y']
                    except Exception as e:
                        print(f'Error processing object {obj_index} coordinates for frame {frame_seq_number} in recording {recording_id}.');

                else:
                    for coord in ['x1', 'y1', 'x2', 'y2']:
                        frame_data[f'object_{obj_index}_{coord}'] = None

            expanded_data.append(frame_data)

    df = pd.DataFrame(expanded_data)
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract frame features from frame coordinates')
    parser.add_argument('--frame_coordinate_id', nargs='+', type=int, default=[1018],
                        help='id values of the frame_coordinate table')
    args = parser.parse_args()

    db_util = DatabaseUtil()
    landmarks =['RIGHT_ELBOW','RIGHT_WRIST','RIGHT_MIDDLE_FINGER_MCP']

    # Convert list of IDs from argparse to a tuple, as pyodbc expects a tuple for parameterized queries
    frame_coordinate_entry_ids = tuple(args.frame_coordinate_id)

    # Fetch frame coordinates data
    frame_coordinates_data = fetch_frame_coordinates(frame_coordinate_entry_ids)

    df = transform_data_to_dataframe(frame_coordinates_data)
    max_frame_seq = df['frame_seq_number'].max()
    frame_slider = widgets.IntSlider(value=0, min=0, max=max_frame_seq, step=1, description='Frame Seq Number:',
                                     continuous_update=False)

    widgets.interactive(plot_landmarks_for_frame, frame_seq_number=21)
    print(df)







