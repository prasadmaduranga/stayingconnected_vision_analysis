import cv2
import mediapipe as mp
import argparse
import os
import json
import sys
import math
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from body_landmarks import BodyLandmark
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from src.util.base_util import log_error

sys.path.append('../')
from util import DatabaseUtil
from util.feature_util import *

from util  import calculate_angle,calculate_bbox,bbox_iou,calculate_distance,calculate_object_center,calculate_line_equation
import pandas as pd
import numpy as np



project_path = '/Users/prasadmaduranga/higher_studies/research/Stroke research/Projects/Staying connected Project/My Projects/stayingconnected_vision_analysis'

# HAND_BOUNDING_BOX_IOU


def calculate_speed(df, landmark_name, frame_rate):
    """
    Calculate the speed of a landmark between consecutive frames.
    Speed is calculated as the Euclidean distance between the landmark positions in consecutive frames,
    divided by the time between frames, based on the frame rate.
    """
    speeds = [None] * len(df)
    for i in range(1, len(df)):
        try:
            if df.iloc[i]['recording_id'] == df.iloc[i - 1]['recording_id']:  # Ensure same recording
                if pd.isna(df.iloc[i][f'{landmark_name}_X']) or pd.isna(df.iloc[i - 1][f'{landmark_name}_X']):
                    continue

                dx = df.iloc[i][f'{landmark_name}_X'] - df.iloc[i - 1][f'{landmark_name}_X']
                dy = df.iloc[i][f'{landmark_name}_Y'] - df.iloc[i - 1][f'{landmark_name}_Y']
                distance = np.sqrt(dx ** 2 + dy ** 2)
                time = 1 / frame_rate
                speed = distance / time
                speeds[i] = round(speed, 4)
        except Exception as e:
            log_error('calculate_speed', f"Error at frame {i}: {str(e)}")
    return speeds

def calculate_grip_aperture(df):
    # Initialize lists to hold the grip aperture values for each frame
    left_grip_apertures = [None] * len(df)
    right_grip_apertures = [None] * len(df)

    for i in range(len(df)):
        try:
            # Calculate PGA for the left hand
            if pd.notna(df.loc[i, 'LEFT_THUMB_TIP_X']) and pd.notna(df.loc[i, 'LEFT_INDEX_FINGER_TIP_X']):
                left_thumb_x, left_thumb_y, left_thumb_z = df.loc[i, 'LEFT_THUMB_TIP_X'], df.loc[i, 'LEFT_THUMB_TIP_Y'], df.loc[i, 'LEFT_THUMB_TIP_Z']
                left_index_x, left_index_y, left_index_z = df.loc[i, 'LEFT_INDEX_FINGER_TIP_X'], df.loc[i, 'LEFT_INDEX_FINGER_TIP_Y'], df.loc[i, 'LEFT_INDEX_FINGER_TIP_Z']
                left_grip_apertures[i] = round(np.sqrt((left_thumb_x - left_index_x) ** 2 + (left_thumb_y - left_index_y) ** 2 + (left_thumb_z - left_index_z) ** 2), 4)

            # Calculate PGA for the right hand
            if pd.notna(df.loc[i, 'RIGHT_THUMB_TIP_X']) and pd.notna(df.loc[i, 'RIGHT_INDEX_FINGER_TIP_X']):
                right_thumb_x, right_thumb_y, right_thumb_z = df.loc[i, 'RIGHT_THUMB_TIP_X'], df.loc[i, 'RIGHT_THUMB_TIP_Y'], df.loc[i, 'RIGHT_THUMB_TIP_Z']
                right_index_x, right_index_y, right_index_z = df.loc[i, 'RIGHT_INDEX_FINGER_TIP_X'], df.loc[i, 'RIGHT_INDEX_FINGER_TIP_Y'], df.loc[i, 'RIGHT_INDEX_FINGER_TIP_Z']
                right_grip_apertures[i] = round(np.sqrt((right_thumb_x - right_index_x) ** 2 + (right_thumb_y - right_index_y) ** 2 + (right_thumb_z - right_index_z) ** 2), 4)
        except Exception as e:
            log_error('calculate_grip_aperture', f"Error at frame {i}: {str(e)}")
    return {'LEFT_GRIP_APERTURE': left_grip_apertures, 'RIGHT_GRIP_APERTURE': right_grip_apertures}


def calculate_wrist_flexion_extension(df):
    left_wrist_angles = [None] * len(df)
    right_wrist_angles = [None] * len(df)

    for i in range(len(df)):
        try:
            # Check if the necessary coordinates are present for the left hand
            if pd.notna(df.loc[i, 'LEFT_ELBOW_X']) and pd.notna(df.loc[i, 'LEFT_WRIST_X']) and pd.notna(df.loc[i, 'LEFT_MIDDLE_FINGER_MCP_X']):
                elbow_left = np.array([df.loc[i, 'LEFT_ELBOW_X'], df.loc[i, 'LEFT_ELBOW_Y'], df.loc[i, 'LEFT_ELBOW_Z']])
                wrist_left = np.array([df.loc[i, 'LEFT_WRIST_X'], df.loc[i, 'LEFT_WRIST_Y'], df.loc[i, 'LEFT_WRIST_Z']])
                mcp_left = np.array([df.loc[i, 'LEFT_MIDDLE_FINGER_MCP_X'], df.loc[i, 'LEFT_MIDDLE_FINGER_MCP_Y'], df.loc[i, 'LEFT_MIDDLE_FINGER_MCP_Z']])
                left_wrist_angles[i] = round(calculate_angle(elbow_left, wrist_left, mcp_left), 4)

            # Check if the necessary coordinates are present for the right hand
            if pd.notna(df.loc[i, 'RIGHT_ELBOW_X']) and pd.notna(df.loc[i, 'RIGHT_WRIST_X']) and pd.notna(df.loc[i, 'RIGHT_MIDDLE_FINGER_MCP_X']):
                elbow_right = np.array([df.loc[i, 'RIGHT_ELBOW_X'], df.loc[i, 'RIGHT_ELBOW_Y'], df.loc[i, 'RIGHT_ELBOW_Z']])
                wrist_right = np.array([df.loc[i, 'RIGHT_WRIST_X'], df.loc[i, 'RIGHT_WRIST_Y'], df.loc[i, 'RIGHT_WRIST_Z']])
                mcp_right = np.array([df.loc[i, 'RIGHT_MIDDLE_FINGER_MCP_X'], df.loc[i, 'RIGHT_MIDDLE_FINGER_MCP_Y'], df.loc[i, 'RIGHT_MIDDLE_FINGER_MCP_Z']])
                right_wrist_angles[i] = round(calculate_angle(elbow_right, wrist_right, mcp_right), 4)
        except Exception as e:
            log_error('calculate_wrist_flexion_extension', f"Error at frame {i}: {str(e)}")
    return {'LEFT_WRIST_ANGLE': left_wrist_angles, 'RIGHT_WRIST_ANGLE': right_wrist_angles}

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


def calculate_hand_bounding_box_iou(df):
    """
    Calculate the Intersection Over Union (IOU) for the bounding boxes of the left and right hands.
    """
    # Initialize IOU column with zeros
    iou_values = [None] * len(df)

    for i in range(len(df)):
        try:
            # Define bounding boxes for left and right hands
            # Assuming landmarks are numbered and correspond to specific fingers and the wrist
            left_hand_points = [(df.iloc[i][f'LEFT_{landmark}_X'], df.iloc[i][f'LEFT_{landmark}_Y'])
                                for landmark in
                                ['WRIST', 'THUMB_TIP', 'INDEX_FINGER_TIP', 'MIDDLE_FINGER_TIP', 'RING_FINGER_TIP',
                                 'PINKY_TIP']
                                if not pd.isna(df.iloc[i][f'LEFT_{landmark}_X'])]

            right_hand_points = [(df.iloc[i][f'RIGHT_{landmark}_X'], df.iloc[i][f'RIGHT_{landmark}_Y'])
                                 for landmark in
                                 ['WRIST', 'THUMB_TIP', 'INDEX_FINGER_TIP', 'MIDDLE_FINGER_TIP', 'RING_FINGER_TIP',
                                  'PINKY_TIP']
                                 if not pd.isna(df.iloc[i][f'RIGHT_{landmark}_X'])]

            if not left_hand_points or not right_hand_points:
                iou_values[i] = 0  # No IOU calculation possible if any hand's points are missing
                continue # Skip if any hand's points are missing

            # Calculate bounding boxes
            left_hand_bbox = calculate_bbox(left_hand_points)
            right_hand_bbox = calculate_bbox(right_hand_points)

            # Calculate IOU
            iou_values[i] = round(bbox_iou(left_hand_bbox, right_hand_bbox), 4)
        except Exception as e:
            log_error('calculate_hand_bounding_box_iou', f"{left_hand_points}, {right_hand_points} \n Error at index {i}: {str(e)}")

    return iou_values

def calculate_acceleration(speeds, frame_rate=10):
    """
    Calculate acceleration from speed values for consecutive frames.
    speeds: List of speeds for consecutive frames.
    frame_rate: Frame rate of the video.
    """
    accelerations = [None]  # The first frame cannot have an acceleration value
    for i in range(1, len(speeds)):
        try:
            if speeds[i] is not None and speeds[i - 1] is not None:
                accel = (speeds[i] - speeds[i - 1]) * frame_rate
                accelerations.append(round(accel, 4))
            else:
                accelerations.append(None)
        except Exception as e:
            log_error('calculate_acceleration', f"Error at index {i}: {str(e)}")
    return accelerations


def calculate_jerk(accelerations, frame_rate=10):
    """
    Calculate jerk from acceleration values for consecutive frames.
    accelerations: List of accelerations for consecutive frames.
    frame_rate: Frame rate of the video.
    """
    jerks = [None, None]  # The first two frames cannot have a jerk value
    for i in range(2, len(accelerations)):
        try:
            if accelerations[i] is not None and accelerations[i-1] is not None:
                jerk = (accelerations[i] - accelerations[i-1]) * frame_rate
                jerks.append(round(jerk, 4))
            else:
                jerks.append(None)
        except Exception as e:
            log_error('calculate_jerk', f"Error at index {i}: {str(e)}")
    return jerks

def calculate_elbow_flexion_angle(df):
    """
    Calculate the elbow flexion angle for both arms.
    """
    left_elbow_angles = [None] * len(df)
    right_elbow_angles = [None] * len(df)

    for i in range(len(df)):
        # Calculate for the left arm
        if pd.notna(df.loc[i, 'LEFT_SHOULDER_X']) and pd.notna(df.loc[i, 'LEFT_ELBOW_X']) and pd.notna(df.loc[i, 'LEFT_WRIST_X']):
            shoulder_left = np.array([df.loc[i, 'LEFT_SHOULDER_X'], df.loc[i, 'LEFT_SHOULDER_Y'], df.loc[i, 'LEFT_SHOULDER_Z']])
            elbow_left = np.array([df.loc[i, 'LEFT_ELBOW_X'], df.loc[i, 'LEFT_ELBOW_Y'], df.loc[i, 'LEFT_ELBOW_Z']])
            wrist_left = np.array([df.loc[i, 'LEFT_WRIST_X'], df.loc[i, 'LEFT_WRIST_Y'], df.loc[i, 'LEFT_WRIST_Z']])
            left_elbow_angles[i] = round(calculate_angle(shoulder_left, elbow_left, wrist_left), 4)

        # Calculate for the right arm
        if pd.notna(df.loc[i, 'RIGHT_SHOULDER_X']) and pd.notna(df.loc[i, 'RIGHT_ELBOW_X']) and pd.notna(df.loc[i, 'RIGHT_WRIST_X']):
            shoulder_right = np.array([df.loc[i, 'RIGHT_SHOULDER_X'], df.loc[i, 'RIGHT_SHOULDER_Y'], df.loc[i, 'RIGHT_SHOULDER_Z']])
            elbow_right = np.array([df.loc[i, 'RIGHT_ELBOW_X'], df.loc[i, 'RIGHT_ELBOW_Y'], df.loc[i, 'RIGHT_ELBOW_Z']])
            wrist_right = np.array([df.loc[i, 'RIGHT_WRIST_X'], df.loc[i, 'RIGHT_WRIST_Y'], df.loc[i, 'RIGHT_WRIST_Z']])
            right_elbow_angles[i] = round(calculate_angle(shoulder_right, elbow_right, wrist_right), 4)

    return {'LEFT_ELBOW_ANGLE': left_elbow_angles, 'RIGHT_ELBOW_ANGLE': right_elbow_angles}


def calculate_shoulder_abduction_angle(df):
    left_shoulder_angles = [None] * len(df)
    right_shoulder_angles = [None] * len(df)

    for i in range(len(df)):
        # Calculate for the left shoulder
        if all(pd.notna(df.loc[i, f'{side}_SHOULDER_X']) for side in ['LEFT', 'RIGHT']) and pd.notna(df.loc[i, 'LEFT_ELBOW_X']):
            shoulder_left = np.array([df.loc[i, 'LEFT_SHOULDER_X'], df.loc[i, 'LEFT_SHOULDER_Y'], df.loc[i, 'LEFT_SHOULDER_Z']])
            shoulder_right = np.array([df.loc[i, 'RIGHT_SHOULDER_X'], df.loc[i, 'RIGHT_SHOULDER_Y'], df.loc[i, 'RIGHT_SHOULDER_Z']])
            elbow_left = np.array([df.loc[i, 'LEFT_ELBOW_X'], df.loc[i, 'LEFT_ELBOW_Y'], df.loc[i, 'LEFT_ELBOW_Z']])
            left_shoulder_angles[i] = round(calculate_angle(shoulder_right, shoulder_left, elbow_left), 4)

        # Calculate for the right shoulder
        if all(pd.notna(df.loc[i, f'{side}_SHOULDER_X']) for side in ['LEFT', 'RIGHT']) and pd.notna(df.loc[i, 'RIGHT_ELBOW_X']):
            shoulder_left = np.array([df.loc[i, 'LEFT_SHOULDER_X'], df.loc[i, 'LEFT_SHOULDER_Y'], df.loc[i, 'LEFT_SHOULDER_Z']])
            shoulder_right = np.array([df.loc[i, 'RIGHT_SHOULDER_X'], df.loc[i, 'RIGHT_SHOULDER_Y'], df.loc[i, 'RIGHT_SHOULDER_Z']])
            elbow_right = np.array([df.loc[i, 'RIGHT_ELBOW_X'], df.loc[i, 'RIGHT_ELBOW_Y'], df.loc[i, 'RIGHT_ELBOW_Z']])
            right_shoulder_angles[i] = round(calculate_angle(shoulder_left, shoulder_right, elbow_right), 4)

    return {'LEFT_SHOULDER_ABDUCTION_ANGLE': left_shoulder_angles, 'RIGHT_SHOULDER_ABDUCTION_ANGLE': right_shoulder_angles}

def calculate_finger_features(df, hand_prefix):
    """
    Calculate finger curvature features for a given hand.
    hand_prefix: 'LEFT_' or 'RIGHT_' indicating the hand.
    """
    # Initialize dictionaries to hold the results
    angles_wrist_mcp_pip = {finger: [None] * len(df) for finger in ['INDEX_FINGER', 'MIDDLE_FINGER', 'RING_FINGER', 'PINKY']}
    angles_mcp_pip_dip = {finger: [None] * len(df) for finger in ['INDEX_FINGER', 'MIDDLE_FINGER', 'RING_FINGER', 'PINKY']}
    angles_pip_dip_tip = {finger: [None] * len(df) for finger in ['INDEX_FINGER', 'MIDDLE_FINGER', 'RING_FINGER', 'PINKY']}
    angles_wrist_mcp_tip = {finger: [None] * len(df) for finger in ['INDEX_FINGER', 'MIDDLE_FINGER', 'RING_FINGER', 'PINKY']}
    ratios_mcp_tip = {finger: [None] * len(df) for finger in ['INDEX_FINGER', 'MIDDLE_FINGER', 'RING_FINGER', 'PINKY']}


    for i in range(len(df)):
        try:
            for finger in ['INDEX_FINGER', 'MIDDLE_FINGER', 'RING_FINGER', 'PINKY']:
                # Construct joint names
                wrist = np.array([df.loc[i, f'{hand_prefix}WRIST_X'], df.loc[i, f'{hand_prefix}WRIST_Y'], df.loc[i, f'{hand_prefix}WRIST_Z']])
                mcp = np.array([df.loc[i, f'{hand_prefix}{finger}_MCP_X'], df.loc[i, f'{hand_prefix}{finger}_MCP_Y'], df.loc[i, f'{hand_prefix}{finger}_MCP_Z']])
                pip = np.array([df.loc[i, f'{hand_prefix}{finger}_PIP_X'], df.loc[i, f'{hand_prefix}{finger}_PIP_Y'], df.loc[i, f'{hand_prefix}{finger}_PIP_Z']])
                dip = np.array([df.loc[i, f'{hand_prefix}{finger}_DIP_X'], df.loc[i, f'{hand_prefix}{finger}_DIP_Y'], df.loc[i, f'{hand_prefix}{finger}_DIP_Z']])
                tip = np.array([df.loc[i, f'{hand_prefix}{finger}_TIP_X'], df.loc[i, f'{hand_prefix}{finger}_TIP_Y'], df.loc[i, f'{hand_prefix}{finger}_TIP_Z']])

                # if any of the joints is missing, continue to next finger
                if any(pd.isna(joint).any() for joint in [wrist, mcp, pip, dip, tip]):
                    ratios_mcp_tip[finger][i] = None
                    continue

                # Calculate angles
                angles_wrist_mcp_pip[finger][i] = calculate_angle(wrist, mcp, pip)
                angles_mcp_pip_dip[finger][i] = calculate_angle(mcp, pip, dip)
                angles_pip_dip_tip[finger][i] = calculate_angle(pip, dip, tip)
                angles_wrist_mcp_tip[finger][i] = calculate_angle(wrist, mcp, tip)

                # Calculate distances and ratio
                dist_mcp_tip = calculate_distance(mcp, tip)
                total_dist = calculate_distance(mcp, pip) + calculate_distance(pip, dip) + calculate_distance(dip, tip)
                ratios_mcp_tip[finger][i] = dist_mcp_tip / total_dist if total_dist else None
        except Exception as e:
            log_error('calculate_finger_features', f"Error at frame {i}: {str(e)}")

    return angles_wrist_mcp_pip, angles_mcp_pip_dip, angles_pip_dip_tip, angles_wrist_mcp_tip, ratios_mcp_tip

def calculate_object_speed(centers, frame_rate=10):
    """
    Calculate the speed of an object between consecutive frames based on its center coordinates.
    centers: List of tuples containing the center coordinates (x, y) of the object for each frame.
    frame_rate: Frame rate of the video.
    """
    speeds = [None]*len(df)  # The first frame cannot have a speed value
    for i in range(1, len(centers)):
        if centers[i] is not None and centers[i-1] is not None:
            dx = centers[i][0] - centers[i-1][0]
            dy = centers[i][1] - centers[i-1][1]
            distance = np.sqrt(dx**2 + dy**2)
            time = 1 / frame_rate
            speed = distance / time
            speeds[i] = round(speed, 4)

    return speeds

def calculate_average_center(df, column_prefix, first_n=10):
    """Calculate the average center for the first `n` frames."""
    x_coords = df[f'{column_prefix}_X'][:first_n]
    y_coords = df[f'{column_prefix}_Y'][:first_n]
    return np.mean(x_coords), np.mean(y_coords)


def calculate_average_center_from_bbox(df, object_prefix, first_n=10):
    """
    Calculate the average center of an object's bounding box for the first `n` frames.
    object_prefix: Prefix of the object column names, e.g., 'object_1'.
    first_n: Number of initial frames to consider for averaging.
    """
    # Extract the bounding box coordinates for the first `n` frames
    x1_coords = df[f'{object_prefix}_x1'][:first_n]
    y1_coords = df[f'{object_prefix}_y1'][:first_n]
    x2_coords = df[f'{object_prefix}_x2'][:first_n]
    y2_coords = df[f'{object_prefix}_y2'][:first_n]

    # Calculate the center for each bounding box
    centers_x = (x1_coords + x2_coords) / 2
    centers_y = (y1_coords + y2_coords) / 2

    # Calculate and return the average center coordinates
    avg_center_x = np.mean(centers_x)
    avg_center_y = np.mean(centers_y)

    return avg_center_x, avg_center_y


def calculate_deviation_from_path(df, m, b):
    """
    Calculate the deviation of object centers from the defined straight path.
    The object center is determined by the average of the bounding box's corner coordinates.
    """
    deviations = []

    # if m or b of nan ,None or empty, return None
    if pd.isna(m) or pd.isna(b) or not m or not b:
        return [None]*len(df)

    for i in range(len(df)):
        # Calculate the center of the bounding box for the object in each frame
        x1, y1 = df.loc[i, 'object_1_x1'], df.loc[i, 'object_1_y1']
        x2, y2 = df.loc[i, 'object_1_x2'], df.loc[i, 'object_1_y2']

        if pd.isna(x1) or pd.isna(y1) or pd.isna(x2) or pd.isna(y2):
            deviations.append(None)
            continue

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # Calculate deviation for the object center from the straight path
        d = abs(m * center_x - center_y + b) / np.sqrt(m ** 2 + 1)
        deviations.append(round(d, 4))

    return deviations

def calculate_hand_object_iou(df):
    left_hand_iou = [None] * len(df)
    right_hand_iou = [None] * len(df)

    for i in range(len(df)):
        if any(pd.isna(df.loc[i, ['object_1_x1', 'object_1_y1', 'object_1_x2', 'object_1_y2']])):
            continue
        # Define bounding box for the object
        object_bbox = (df.loc[i, 'object_1_x1'], df.loc[i, 'object_1_y1'], df.loc[i, 'object_1_x2'], df.loc[i, 'object_1_y2'])

        # Define bounding boxes for left and right hands
        left_hand_points = [(df.loc[i, f'LEFT_{landmark}_X'], df.loc[i, f'LEFT_{landmark}_Y'])
                            for landmark in ['WRIST', 'THUMB_TIP', 'INDEX_FINGER_TIP', 'MIDDLE_FINGER_TIP', 'RING_FINGER_TIP', 'PINKY_TIP']
                            if not pd.isna(df.loc[i, f'LEFT_{landmark}_X'])]

        right_hand_points = [(df.loc[i, f'RIGHT_{landmark}_X'], df.loc[i, f'RIGHT_{landmark}_Y'])
                             for landmark in ['WRIST', 'THUMB_TIP', 'INDEX_FINGER_TIP', 'MIDDLE_FINGER_TIP', 'RING_FINGER_TIP', 'PINKY_TIP']
                             if not pd.isna(df.loc[i, f'RIGHT_{landmark}_X'])]

        if left_hand_points:
            left_hand_bbox = calculate_bbox(left_hand_points)
            left_hand_iou[i] = bbox_iou(object_bbox, left_hand_bbox)

        if right_hand_points:
            right_hand_bbox = calculate_bbox(right_hand_points)
            right_hand_iou[i] = bbox_iou(object_bbox, right_hand_bbox)

    return left_hand_iou, right_hand_iou

def calculate_features(df):
    """
    Calculate various features for the dataframe, including speeds of selected landmarks,
    and return these as a new DataFrame with the capability to add more features later.
    """
    if not df.empty and 'frame_rate' in df.columns:
        # Extract identifying information and create a base DataFrame
        base_data = {
            'frame_coordinate_id': df['frame_coordinate_id'],
            'recording_id': df['recording_id'],
            'frame_rate': df['frame_rate'],
            'frame_seq_number': df['frame_seq_number']
        }
        feature_df = pd.DataFrame(base_data)

        # Assuming constant frame rate for simplicity
        frame_rate = df['frame_rate'].iloc[0]

        # Calculate speeds for specified landmarks and add them to the feature DataFrame
        for landmark in ['RIGHT_WRIST', 'RIGHT_ELBOW', 'LEFT_WRIST', 'LEFT_ELBOW']:
            speed_feature_name = f'{landmark}_SPEED'
            acceleration_feature_name = f'{landmark}_ACCELERATION'
            jerk_feature_name = f'{landmark}_JERK'
            feature_df[speed_feature_name] = calculate_speed(df, landmark, frame_rate)
            feature_df[acceleration_feature_name] = calculate_acceleration(feature_df[speed_feature_name], frame_rate)
            feature_df[jerk_feature_name] = calculate_jerk(feature_df[acceleration_feature_name], frame_rate)


        # 1. grip aperture (PGA) is the distance between the thumb and index finger
        # Calculate grip aperture for each frame and for each hand and add it to the feature DataFrame feature_df
        grip_apertures = calculate_grip_aperture(df)
        feature_df['LEFT_GRIP_APERTURE'] = grip_apertures['LEFT_GRIP_APERTURE']
        feature_df['RIGHT_GRIP_APERTURE'] = grip_apertures['RIGHT_GRIP_APERTURE']

        # 2. Wrist flexion / extension angle
        # calculated by the angle between elbow joint, wrist and middle finger mcps
        # Calculate wrist flexion/extension angle for each frame and each hand add it to the feature DataFrame feature_df
        wrist_angles = calculate_wrist_flexion_extension(df)
        feature_df['LEFT_WRIST_FLEXION_EXTENSION_ANGLE'] = wrist_angles['LEFT_WRIST_ANGLE']
        feature_df['RIGHT_WRIST_FLEXION_EXTENSION_ANGLE'] = wrist_angles['RIGHT_WRIST_ANGLE']

        # 3. haand iou
        # compensation > iou between hands
        #  first define a bounding box for the each hand. use the landmarks of the hand. Consider the joints including wrist and joints of each finger
        #  calculate the iou between the bounding boxes of the left and right hands
        #  calculate ios_hands each frame and add to the DataFrame feature_df
        feature_df['HAND_BOUNDING_BOX_IOU'] = calculate_hand_bounding_box_iou(df)

        # 4. acceleration
        # calculate the acceleration of the wrist and elbow joints. Use the speed values calculated in step 1
        # calculate the acceleration of the wrist and elbow joints for each frame and each hand and add it to the feature DataFrame feature_df



        # 5. jerk
        # calculate the jerk of the wrist and elbow joints. Use the acceleration values calculated in step 4
        # calculate the jerk of the wrist and elbow joints for each frame and each hand and add it to the feature DataFrame feature_df


            # 6. elbow flexion angle
            # calculate the angle between the upper arm and the forearm. Angle should be derived by shoulder joint elbow joint and wrist joint
            # calculate for both hands and add to the feature DataFrame feature_df

        elbow_flexion_angles = calculate_elbow_flexion_angle(df)
        feature_df['LEFT_ELBOW_FLEXION_ANGLE'] = elbow_flexion_angles['LEFT_ELBOW_ANGLE']
        feature_df['RIGHT_ELBOW_FLEXION_ANGLE'] = elbow_flexion_angles['RIGHT_ELBOW_ANGLE']

        # 7. shoulder abduction angle
        # calculate the angle between the line connecting the shoulder joints and the elbow joint of each arm
        # calculate for both hands and add to the feature DataFrame feature_df
        shoulder_abduction_angles = calculate_shoulder_abduction_angle(df)
        feature_df['LEFT_SHOULDER_ABDUCTION_ANGLE'] = shoulder_abduction_angles['LEFT_SHOULDER_ABDUCTION_ANGLE']
        feature_df['RIGHT_SHOULDER_ABDUCTION_ANGLE'] = shoulder_abduction_angles['RIGHT_SHOULDER_ABDUCTION_ANGLE']

        # 8. curvature of the each finger except thumb
        # calculate the curvature of the each finger except thumb. Use the landmarks (MCP,PIP,DIP,TIP) of the finger joints
        # calculate for each frame and each hand and add to the feature DataFrame feature_df
        # four fingers, , 4 anglesn each finger, 12 angles, (mcp,tip distance / pip mcp distance, pip dip distance, dip tip distance)
        #  calculate following four angles and distal for each of the four fingers
        # 1. angle between wrist, mcp and pip
        # 2. angle between mcp, pip and dip
        # 3. angle between pip, dip and tip
        # 4. angle between wrist, mcp and tip
        # 5. (distal between mcp and tip)/ (pip mcp distance+ pip dip distance+ dip tip distance)
        # Assuming the function calculate_finger_features is already defined

        left_hand_features = calculate_finger_features(df, 'LEFT_')
        right_hand_features = calculate_finger_features(df, 'RIGHT_')

        feature_names = ['ANGLE_WRIST_MCP_PIP', 'ANGLE_MCP_PIP_DIP', 'ANGLE_PIP_DIP_TIP', 'ANGLE_WRIST_MCP_TIP',
                         'RATIO_MCP_TIP_DISTAL']

        for finger in ['INDEX_FINGER', 'MIDDLE_FINGER', 'RING_FINGER', 'PINKY']:
            for i, feature_name in enumerate(feature_names):
                feature_df[f'LEFT_{finger}_{feature_name}'] = left_hand_features[i][finger]
                feature_df[f'RIGHT_{finger}_{feature_name}'] = right_hand_features[i][finger]

        # ====================

        # Object features
        # 9. speed of object
        # calculate the speed of the object. get the center of the object by considering the coner joint coordinates given. (object_1_x1, object_1_y1, object_1_x2, object_1_y2) are coordinates of the object bounding box
        #  simlarly for the object_2. calculate the speed of the object for each frame and add to the feature DataFrame feature_df (use the speed function)
        frame_rate = 10  # Adjust based on your actual frame rate

        # Calculate center points for each object in each frame
        object_1_centers = [(calculate_object_center(df['object_1_x1'][i], df['object_1_y1'][i], df['object_1_x2'][i],
                                                     df['object_1_y2'][i]))
                            if not pd.isna(df['object_1_x1'][i]) else None for i in range(len(df))]
        object_2_centers = [(calculate_object_center(df['object_2_x1'][i], df['object_2_y1'][i], df['object_2_x2'][i],
                                                     df['object_2_y2'][i]))
                            if not pd.isna(df['object_2_x1'][i]) else None for i in range(len(df))]

        # Calculate speed for each object
        feature_df['object_1_speed'] = calculate_object_speed(object_1_centers, frame_rate)
        feature_df['object_2_speed'] = calculate_object_speed(object_2_centers, frame_rate)

        # 10. trajectory deviation to the shortest path
        # calculate the deviation of the object trajectory to the shortest path between the start and end points.
        # get the first ten frames and averaage the coordinates of the object center. This is the starting point
        # get the first ten frames and averaage the coordinates of the mouth center. This is the end point. Mouth center is taaken by geting the center of MOUTH_LEFT, MOUTH_RIGHT of df
        # draw a straight line between the start and end points. calculate the deviation of the object trajectory to the shortest path (get the perpendicular distance)at each frame, add to the feature DataFrame feature_df
        object_start = calculate_average_center_from_bbox(df, 'object_1', first_n=10)  # Assuming object_1_center is calculated and stored
        mouth_start, mouth_end = calculate_average_center(df, 'MOUTH_LEFT'), calculate_average_center(df, 'MOUTH_RIGHT')
        mouth_center = ((mouth_start[0] + mouth_end[0]) / 2, (mouth_start[1] + mouth_end[1]) / 2)
        m, b = calculate_line_equation(object_start, mouth_center)
        feature_df['object_1_trajectory_deviation'] = calculate_deviation_from_path(df, m, b)


        # 11. object hand iou
        # calculate the iou between the object bounding box and the bounding box of the each hand
        # first define a bounding box for the object. use the coordinates of the corner
        # calculate the iou between the object bounding box and the bounding box of the each hand and store in the feature DataFrame feature_df
        feature_df['object_1_left_hand_iou'], feature_df['object_1_right_hand_iou'] = calculate_hand_object_iou(df)

        return feature_df

    else:
        # Return an empty DataFrame if the input df is empty or lacks frame_rate column
        return pd.DataFrame()


def visualize_features(features_df):
    """
    Visualize the specified features on a line graph.
    Each feature is plotted as a separate line.
    NaN or None values are replaced by 0.
    """
    # Replace NaN or None with 0
    features_df.fillna(0, inplace=True)

    plt.figure(figsize=(15, 10))  # Set the figure size for better readability

    # Plot each feature as a separate line
    for column in features_df.columns:
        # Skip non-feature columns
        if column in ['frame_coordinate_id', 'recording_id', 'frame_rate', 'frame_seq_number']:
            continue
        plt.plot(features_df['frame_seq_number'], features_df[column], label=column)

    plt.xlabel('Frame Sequence Number')
    plt.ylabel('Speed')
    plt.title('Feature Speeds Over Time')
    plt.legend()
    plt.show()


def smooth_data_moving_average(df, window_size=3):
    """
    Smooths each feature in the DataFrame using a moving average.
    `window_size` determines the size of the moving window over which the average is computed.
    """
    smoothed_df = df.copy()
    feature_columns = [col for col in df.columns if
                       col not in ['frame_coordinate_id', 'recording_id', 'frame_rate', 'frame_seq_number']]

    for column in feature_columns:
        smoothed_df[column] = df[column].rolling(window=window_size, min_periods=1, center=True).mean()

    return smoothed_df

# def write_features_to_db(df, db_util):
#     """
#     Writes the features from the DataFrame to the database, converting NaN values to NULL.
#     """
#     # Prepare the INSERT query template
#     insert_query = """
#     INSERT INTO VisionAnalysis.dbo.frame_features
#     (id, frame_coordinate_id, frame_seq_number, recording_id, frame_rate, right_wrist_speed, right_elbow_speed, left_wrist_speed, left_elbow_speed, hand_bounding_box_iou)
#     VALUES (NEWID(), ?, ?, ?, ?, ?, ?, ?, ?, ?);
#     """
#
#     insert_query = """
#        INSERT INTO VisionAnalysis.dbo.frame_features
#        (id,frame_coordinate_id, recording_id, frame_rate, file_name)
#        VALUES (NEWID(), ?, ?, ?, ?);
#        """
#
#     # Iterate over the DataFrame rows
#     for index, row in df.iterrows():
#         # Prepare the parameters for the current row, replacing NaN with None
#         params = (
#             row['frame_coordinate_id'],
#             row['frame_seq_number'],
#             row['recording_id'],
#             row['frame_rate'],
#             None if pd.isna(row['RIGHT_WRIST_SPEED']) else row['RIGHT_WRIST_SPEED'],
#             None if pd.isna(row['RIGHT_ELBOW_SPEED']) else row['RIGHT_ELBOW_SPEED'],
#             None if pd.isna(row['LEFT_WRIST_SPEED']) else row['LEFT_WRIST_SPEED'],
#             None if pd.isna(row['LEFT_ELBOW_SPEED']) else row['LEFT_ELBOW_SPEED'],
#             None if pd.isna(row['HAND_BOUNDING_BOX_IOU']) else row['HAND_BOUNDING_BOX_IOU']
#         )
#
#         # Execute the query with the current row's parameters
#         db_util.insert_data(insert_query, params)

import pandas as pd
import pandas as pd

import pandas as pd

def write_features_to_db(df, db_util):
    """
    Writes the features from the DataFrame to the database, converting NaN values to NULL.
    """
    # Prepare the INSERT query template
    insert_query = """
    INSERT INTO VisionAnalysis.dbo.frame_features (
        id, frame_coordinate_id, frame_seq_number, recording_id, frame_rate, 
        right_wrist_speed, right_wrist_acceleration, right_wrist_jerk, 
        right_elbow_speed, right_elbow_acceleration, right_elbow_jerk, 
        left_wrist_speed, left_wrist_acceleration, left_wrist_jerk, 
        left_elbow_speed, left_elbow_acceleration, left_elbow_jerk, 
        left_grip_aperture, right_grip_aperture, 
        left_wrist_flexion_extension_angle, right_wrist_flexion_extension_angle, hand_bounding_box_iou, 
        left_elbow_flexion_angle, right_elbow_flexion_angle, 
        left_shoulder_abduction_angle, right_shoulder_abduction_angle, 
        left_index_finger_angle_wrist_mcp_pip, right_index_finger_angle_wrist_mcp_pip, 
        left_index_finger_angle_mcp_pip_dip, right_index_finger_angle_mcp_pip_dip, 
        left_index_finger_angle_pip_dip_tip, right_index_finger_angle_pip_dip_tip, 
        left_index_finger_angle_wrist_mcp_tip, right_index_finger_angle_wrist_mcp_tip, 
        left_index_finger_ratio_mcp_tip_distal, right_index_finger_ratio_mcp_tip_distal, 
        left_middle_finger_angle_wrist_mcp_pip, right_middle_finger_angle_wrist_mcp_pip, 
        left_middle_finger_angle_mcp_pip_dip, right_middle_finger_angle_mcp_pip_dip, 
        left_middle_finger_angle_pip_dip_tip, right_middle_finger_angle_pip_dip_tip, 
        left_middle_finger_angle_wrist_mcp_tip, right_middle_finger_angle_wrist_mcp_tip, 
        left_middle_finger_ratio_mcp_tip_distal, right_middle_finger_ratio_mcp_tip_distal, 
        left_ring_finger_angle_wrist_mcp_pip, right_ring_finger_angle_wrist_mcp_pip, 
        left_ring_finger_angle_mcp_pip_dip, right_ring_finger_angle_mcp_pip_dip, 
        left_ring_finger_angle_pip_dip_tip, right_ring_finger_angle_pip_dip_tip, 
        left_ring_finger_angle_wrist_mcp_tip, right_ring_finger_angle_wrist_mcp_tip, 
        left_ring_finger_ratio_mcp_tip_distal, right_ring_finger_ratio_mcp_tip_distal, 
        left_pinky_angle_wrist_mcp_pip, right_pinky_angle_wrist_mcp_pip, 
        left_pinky_angle_mcp_pip_dip, right_pinky_angle_mcp_pip_dip, 
        left_pinky_angle_pip_dip_tip, right_pinky_angle_pip_dip_tip, 
        left_pinky_angle_wrist_mcp_tip, right_pinky_angle_wrist_mcp_tip, 
        left_pinky_ratio_mcp_tip_distal, right_pinky_ratio_mcp_tip_distal, 
        object_1_speed, object_2_speed, object_1_trajectory_deviation, 
        object_1_left_hand_iou, object_1_right_hand_iou
    ) VALUES (NEWID(), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """

    # Prepare a list of tuples for the batch insert
    values = []
    # if df is none , log it and return
    if df is None:
        print("Dataframe is None, skipping write features to db")
        return


    for index, row in df.iterrows():
        params = (
            None if pd.isna(row['frame_coordinate_id']) else row['frame_coordinate_id'],
            None if pd.isna(row['frame_seq_number']) else row['frame_seq_number'],
            None if pd.isna(row['recording_id']) else row['recording_id'],
            None if pd.isna(row['frame_rate']) else row['frame_rate'],
            None if pd.isna(row['RIGHT_WRIST_SPEED']) else row['RIGHT_WRIST_SPEED'],
            None if pd.isna(row['RIGHT_WRIST_ACCELERATION']) else row['RIGHT_WRIST_ACCELERATION'],
            None if pd.isna(row['RIGHT_WRIST_JERK']) else row['RIGHT_WRIST_JERK'],
            None if pd.isna(row['RIGHT_ELBOW_SPEED']) else row['RIGHT_ELBOW_SPEED'],
            None if pd.isna(row['RIGHT_ELBOW_ACCELERATION']) else row['RIGHT_ELBOW_ACCELERATION'],
            None if pd.isna(row['RIGHT_ELBOW_JERK']) else row['RIGHT_ELBOW_JERK'],
            None if pd.isna(row['LEFT_WRIST_SPEED']) else row['LEFT_WRIST_SPEED'],
            None if pd.isna(row['LEFT_WRIST_ACCELERATION']) else row['LEFT_WRIST_ACCELERATION'],
            None if pd.isna(row['LEFT_WRIST_JERK']) else row['LEFT_WRIST_JERK'],
            None if pd.isna(row['LEFT_ELBOW_SPEED']) else row['LEFT_ELBOW_SPEED'],
            None if pd.isna(row['LEFT_ELBOW_ACCELERATION']) else row['LEFT_ELBOW_ACCELERATION'],
            None if pd.isna(row['LEFT_ELBOW_JERK']) else row['LEFT_ELBOW_JERK'],
            None if pd.isna(row['LEFT_GRIP_APERTURE']) else row['LEFT_GRIP_APERTURE'],
            None if pd.isna(row['RIGHT_GRIP_APERTURE']) else row['RIGHT_GRIP_APERTURE'],
            None if pd.isna(row['LEFT_WRIST_FLEXION_EXTENSION_ANGLE']) else row['LEFT_WRIST_FLEXION_EXTENSION_ANGLE'],
            None if pd.isna(row['RIGHT_WRIST_FLEXION_EXTENSION_ANGLE']) else row['RIGHT_WRIST_FLEXION_EXTENSION_ANGLE'],
            None if pd.isna(row['HAND_BOUNDING_BOX_IOU']) else row['HAND_BOUNDING_BOX_IOU'],
            None if pd.isna(row['LEFT_ELBOW_FLEXION_ANGLE']) else row['LEFT_ELBOW_FLEXION_ANGLE'],
            None if pd.isna(row['RIGHT_ELBOW_FLEXION_ANGLE']) else row['RIGHT_ELBOW_FLEXION_ANGLE'],
            None if pd.isna(row['LEFT_SHOULDER_ABDUCTION_ANGLE']) else row['LEFT_SHOULDER_ABDUCTION_ANGLE'],
            None if pd.isna(row['RIGHT_SHOULDER_ABDUCTION_ANGLE']) else row['RIGHT_SHOULDER_ABDUCTION_ANGLE'],
            None if pd.isna(row['LEFT_INDEX_FINGER_ANGLE_WRIST_MCP_PIP']) else row['LEFT_INDEX_FINGER_ANGLE_WRIST_MCP_PIP'],
            None if pd.isna(row['RIGHT_INDEX_FINGER_ANGLE_WRIST_MCP_PIP']) else row['RIGHT_INDEX_FINGER_ANGLE_WRIST_MCP_PIP'],
            None if pd.isna(row['LEFT_INDEX_FINGER_ANGLE_MCP_PIP_DIP']) else row['LEFT_INDEX_FINGER_ANGLE_MCP_PIP_DIP'],
            None if pd.isna(row['RIGHT_INDEX_FINGER_ANGLE_MCP_PIP_DIP']) else row['RIGHT_INDEX_FINGER_ANGLE_MCP_PIP_DIP'],
            None if pd.isna(row['LEFT_INDEX_FINGER_ANGLE_PIP_DIP_TIP']) else row['LEFT_INDEX_FINGER_ANGLE_PIP_DIP_TIP'],
            None if pd.isna(row['RIGHT_INDEX_FINGER_ANGLE_PIP_DIP_TIP']) else row['RIGHT_INDEX_FINGER_ANGLE_PIP_DIP_TIP'],
            None if pd.isna(row['LEFT_INDEX_FINGER_ANGLE_WRIST_MCP_TIP']) else row['LEFT_INDEX_FINGER_ANGLE_WRIST_MCP_TIP'],
            None if pd.isna(row['RIGHT_INDEX_FINGER_ANGLE_WRIST_MCP_TIP']) else row['RIGHT_INDEX_FINGER_ANGLE_WRIST_MCP_TIP'],
            None if pd.isna(row['LEFT_INDEX_FINGER_RATIO_MCP_TIP_DISTAL']) else row['LEFT_INDEX_FINGER_RATIO_MCP_TIP_DISTAL'],
            None if pd.isna(row['RIGHT_INDEX_FINGER_RATIO_MCP_TIP_DISTAL']) else row['RIGHT_INDEX_FINGER_RATIO_MCP_TIP_DISTAL'],
            None if pd.isna(row['LEFT_MIDDLE_FINGER_ANGLE_WRIST_MCP_PIP']) else row['LEFT_MIDDLE_FINGER_ANGLE_WRIST_MCP_PIP'],
            None if pd.isna(row['RIGHT_MIDDLE_FINGER_ANGLE_WRIST_MCP_PIP']) else row['RIGHT_MIDDLE_FINGER_ANGLE_WRIST_MCP_PIP'],
            None if pd.isna(row['LEFT_MIDDLE_FINGER_ANGLE_MCP_PIP_DIP']) else row['LEFT_MIDDLE_FINGER_ANGLE_MCP_PIP_DIP'],
            None if pd.isna(row['RIGHT_MIDDLE_FINGER_ANGLE_MCP_PIP_DIP']) else row['RIGHT_MIDDLE_FINGER_ANGLE_MCP_PIP_DIP'],
            None if pd.isna(row['LEFT_MIDDLE_FINGER_ANGLE_PIP_DIP_TIP']) else row['LEFT_MIDDLE_FINGER_ANGLE_PIP_DIP_TIP'],
            None if pd.isna(row['RIGHT_MIDDLE_FINGER_ANGLE_PIP_DIP_TIP']) else row['RIGHT_MIDDLE_FINGER_ANGLE_PIP_DIP_TIP'],
            None if pd.isna(row['LEFT_MIDDLE_FINGER_ANGLE_WRIST_MCP_TIP']) else row['LEFT_MIDDLE_FINGER_ANGLE_WRIST_MCP_TIP'],
            None if pd.isna(row['RIGHT_MIDDLE_FINGER_ANGLE_WRIST_MCP_TIP']) else row['RIGHT_MIDDLE_FINGER_ANGLE_WRIST_MCP_TIP'],
            None if pd.isna(row['LEFT_MIDDLE_FINGER_RATIO_MCP_TIP_DISTAL']) else row['LEFT_MIDDLE_FINGER_RATIO_MCP_TIP_DISTAL'],
            None if pd.isna(row['RIGHT_MIDDLE_FINGER_RATIO_MCP_TIP_DISTAL']) else row['RIGHT_MIDDLE_FINGER_RATIO_MCP_TIP_DISTAL'],
            None if pd.isna(row['LEFT_RING_FINGER_ANGLE_WRIST_MCP_PIP']) else row['LEFT_RING_FINGER_ANGLE_WRIST_MCP_PIP'],
            None if pd.isna(row['RIGHT_RING_FINGER_ANGLE_WRIST_MCP_PIP']) else row['RIGHT_RING_FINGER_ANGLE_WRIST_MCP_PIP'],
            None if pd.isna(row['LEFT_RING_FINGER_ANGLE_MCP_PIP_DIP']) else row['LEFT_RING_FINGER_ANGLE_MCP_PIP_DIP'],
            None if pd.isna(row['RIGHT_RING_FINGER_ANGLE_MCP_PIP_DIP']) else row['RIGHT_RING_FINGER_ANGLE_MCP_PIP_DIP'],
            None if pd.isna(row['LEFT_RING_FINGER_ANGLE_PIP_DIP_TIP']) else row['LEFT_RING_FINGER_ANGLE_PIP_DIP_TIP'],
            None if pd.isna(row['RIGHT_RING_FINGER_ANGLE_PIP_DIP_TIP']) else row['RIGHT_RING_FINGER_ANGLE_PIP_DIP_TIP'],
            None if pd.isna(row['LEFT_RING_FINGER_ANGLE_WRIST_MCP_TIP']) else row['LEFT_RING_FINGER_ANGLE_WRIST_MCP_TIP'],
            None if pd.isna(row['RIGHT_RING_FINGER_ANGLE_WRIST_MCP_TIP']) else row['RIGHT_RING_FINGER_ANGLE_WRIST_MCP_TIP'],
            None if pd.isna(row['LEFT_RING_FINGER_RATIO_MCP_TIP_DISTAL']) else row['LEFT_RING_FINGER_RATIO_MCP_TIP_DISTAL'],
            None if pd.isna(row['RIGHT_RING_FINGER_RATIO_MCP_TIP_DISTAL']) else row['RIGHT_RING_FINGER_RATIO_MCP_TIP_DISTAL'],
            None if pd.isna(row['LEFT_PINKY_ANGLE_WRIST_MCP_PIP']) else row['LEFT_PINKY_ANGLE_WRIST_MCP_PIP'],
            None if pd.isna(row['RIGHT_PINKY_ANGLE_WRIST_MCP_PIP']) else row['RIGHT_PINKY_ANGLE_WRIST_MCP_PIP'],
            None if pd.isna(row['LEFT_PINKY_ANGLE_MCP_PIP_DIP']) else row['LEFT_PINKY_ANGLE_MCP_PIP_DIP'],
            None if pd.isna(row['RIGHT_PINKY_ANGLE_MCP_PIP_DIP']) else row['RIGHT_PINKY_ANGLE_MCP_PIP_DIP'],
            None if pd.isna(row['LEFT_PINKY_ANGLE_PIP_DIP_TIP']) else row['LEFT_PINKY_ANGLE_PIP_DIP_TIP'],
            None if pd.isna(row['RIGHT_PINKY_ANGLE_PIP_DIP_TIP']) else row['RIGHT_PINKY_ANGLE_PIP_DIP_TIP'],
            None if pd.isna(row['LEFT_PINKY_ANGLE_WRIST_MCP_TIP']) else row['LEFT_PINKY_ANGLE_WRIST_MCP_TIP'],
            None if pd.isna(row['RIGHT_PINKY_ANGLE_WRIST_MCP_TIP']) else row['RIGHT_PINKY_ANGLE_WRIST_MCP_TIP'],
            None if pd.isna(row['LEFT_PINKY_RATIO_MCP_TIP_DISTAL']) else row['LEFT_PINKY_RATIO_MCP_TIP_DISTAL'],
            None if pd.isna(row['RIGHT_PINKY_RATIO_MCP_TIP_DISTAL']) else row['RIGHT_PINKY_RATIO_MCP_TIP_DISTAL'],
            None if pd.isna(row['object_1_speed']) else row['object_1_speed'],
            None if pd.isna(row['object_2_speed']) else row['object_2_speed'],
            None if pd.isna(row['object_1_trajectory_deviation']) else row['object_1_trajectory_deviation'],
            None if pd.isna(row['object_1_left_hand_iou']) else row['object_1_left_hand_iou'],
            None if pd.isna(row['object_1_right_hand_iou']) else row['object_1_right_hand_iou']
        )
        values.append(params)

    # Execute batch insert
    db_util.insert_batch(insert_query, values)


def write_df_to_db_as_pickle(df, db_util):
    """
    Serializes the entire DataFrame into a pickle file and writes file metadata to the database.
    """

    pickle_base_path = os.path.join(project_path, 'data/processed/frame_features')
    if not os.path.exists(pickle_base_path):
        os.makedirs(pickle_base_path)

    # Assuming the DataFrame has consistent values for certain columns across all rows
    frame_coordinate_id = df['frame_coordinate_id'].iloc[0]
    recording_id = df['recording_id'].iloc[0]
    frame_rate = df['frame_rate'].iloc[0]

    # Example of generating a unique file name using a timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f"frame_features_{recording_id}_{timestamp}.pkl"
    file_path = os.path.join(pickle_base_path, file_name)

    # Serialize the entire DataFrame into a pickle file
    with open(file_path, 'wb') as file:
        pickle.dump(df, file)

    # Prepare the INSERT query for file metadata
    insert_query = """
    INSERT INTO VisionAnalysis.dbo.frame_feature_files
    (frame_coordinate_id, recording_id, frame_rate, file_name)
    VALUES (?, ?, ?, ?);
    """

    # Prepare parameters for the INSERT query
    params = (
        str(frame_coordinate_id),  # Convert to string if the database expects VARCHAR
        str(recording_id),  # Convert to string if the database expects VARCHAR
        int(frame_rate),  # Convert numpy.int64 to Python int
        file_name
    )

    # Execute the query with the parameters
    db_util.insert_data(insert_query, params)

def read_feature_file(frame_coordinate_id, recording_id, db_util, project_path):
    """
    Fetches the filename from the database for the given frame_coordinate_id and recording_id,
    reads the pickle file, and returns the DataFrame.
    """
    # Define the base path for the pickle files
    pickle_base_path = os.path.join(project_path, 'data/processed/frame_features')

    # Prepare the SQL query to fetch the filename
    fetch_query = """
    SELECT file_name FROM VisionAnalysis.dbo.frame_feature_files
    WHERE frame_coordinate_id = ? AND recording_id = ?
    """

    # Fetch the file name from the database
    file_name = db_util.fetch_data(fetch_query, (frame_coordinate_id, recording_id))

    # Assuming fetch_data returns a list of tuples and we only need the first result
    if file_name and len(file_name) > 0:
        file_name = file_name[0][0]  # Adjust indexing based on actual return structure
        file_path = os.path.join(pickle_base_path, file_name)

        # Load the pickle file into a DataFrame
        with open(file_path, 'rb') as file:
            df = pickle.load(file)

        return df
    else:
        print("File not found for the given identifiers.")
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract frame features from frame coordinates')
    parser.add_argument('--frame_coordinate_id', nargs='+', type=int, default=[],
                        help='id values of the frame_coordinate table')
    args = parser.parse_args()

    db_util = DatabaseUtil()


    # Convert list of IDs from argparse to a tuple, as pyodbc expects a tuple for parameterized queries
    frame_coordinate_entry_ids = tuple(args.frame_coordinate_id)

    # Fetch frame coordinates data
    frame_coordinates_data = fetch_frame_coordinates(frame_coordinate_entry_ids)

    df = transform_data_to_dataframe(frame_coordinates_data)
    df_with_features = None
    try:
        df_with_features = calculate_features(df)
    except Exception as e:
        print(f"Error occurred while calculating features: {e}")


    write_features_to_db(df_with_features, db_util)
    write_df_to_db_as_pickle(df_with_features, db_util)
    # df2 = read_feature_file(1016, 1061, db_util, project_path)
    # print(df2)
    # smoothed_df = smooth_data_moving_average(df_with_features, window_size=4)
    # visualize_features(smoothed_df)






