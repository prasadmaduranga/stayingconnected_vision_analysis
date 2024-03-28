import cv2
import mediapipe as mp
import argparse
import os
import json
import sys
import math
import pandas as pd
import numpy as np
from body_landmarks import BodyLandmark
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

sys.path.append('../')
from util import DatabaseUtil
from util.feature_util import *


# HAND_BOUNDING_BOX_IOU

def calculate_speed(df, landmark_name, frame_rate):
    """
    Calculate the speed of a landmark between consecutive frames.
    Speed is calculated as the Euclidean distance between the landmark positions in consecutive frames,
    divided by the time between frames, based on the frame rate.
    """
    speeds = [None]*len(df)
    for i in range(1, len(df)):
        if df.iloc[i]['recording_id'] == df.iloc[i-1]['recording_id']:  # Ensure same recording
            if pd.isna(df.iloc[i][f'{landmark_name}_X']) or pd.isna(df.iloc[i - 1][f'{landmark_name}_X']):
                continue

            dx = df.iloc[i][f'{landmark_name}_X'] - df.iloc[i-1][f'{landmark_name}_X']
            dy = df.iloc[i][f'{landmark_name}_Y'] - df.iloc[i-1][f'{landmark_name}_Y']
            distance = np.sqrt(dx**2 + dy**2)
            time = 1 / frame_rate
            speed = distance / time
            #round speed to 4 decimal places
            speeds[i] = round(speed, 4)

    # Handle case of first frame in a recording
    return speeds


def fetch_frame_coordinates(ids):
    """
    Fetches frame coordinates for the given IDs from the database.
    """
    placeholders = ', '.join('?' for _ in ids)  # Create placeholders for parameterized query
    query = f"""
    SELECT id, recording_id, frame_rate, coordinates
    FROM frame_coordinates
    WHERE id IN ({placeholders});
    """
    db_util = DatabaseUtil()
    return db_util.fetch_data(query, ids)

def transform_data_to_dataframe(data):
    """
    Transforms and expands the data from the database into a pandas DataFrame.
    Each frame's coordinates from the JSON array are expanded into separate rows.
    """
    # Initialize an empty list to hold the expanded data
    expanded_data = []

    # Iterate over each record from the database
    for record in data:
        frame_coordinate_id, recording_id, frame_rate, coordinates_json = record
        coordinates = json.loads(coordinates_json)  # Parse the JSON string into a Python list

        # Iterate over each frame's coordinates in the JSON array
        for frame_seq_number, frame_coordinates in enumerate(coordinates):
            # Initialize a dictionary for the current frame's data
            frame_data = {
                'frame_coordinate_id': frame_coordinate_id,
                'recording_id': recording_id,
                'frame_rate': frame_rate,
                'frame_seq_number': frame_seq_number,
            }
            # Add each landmark's coordinates to the frame data
            for landmark in BodyLandmark:
                if landmark.value < len(frame_coordinates):
                    landmark_data = frame_coordinates[landmark.value]
                    if landmark_data is not None:
                        frame_data[f'{landmark.name}_X'] = landmark_data.get('x', None)
                        frame_data[f'{landmark.name}_Y'] = landmark_data.get('y', None)
                        frame_data[f'{landmark.name}_Z'] = landmark_data.get('z', None)
                    else:
                        # If landmark_data is None, set the landmark values in frame_data to None
                        frame_data[f'{landmark.name}_X'] = None
                        frame_data[f'{landmark.name}_Y'] = None
                        frame_data[f'{landmark.name}_Z'] = None
                else:
                    # If the landmark does not exist in the frame_coordinates, set values to None
                    frame_data[f'{landmark.name}_X'] = None
                    frame_data[f'{landmark.name}_Y'] = None
                    frame_data[f'{landmark.name}_Z'] = None

            # Append the current frame's data to the expanded data list
            expanded_data.append(frame_data)

    # Convert the expanded data list into a pandas DataFrame
    df = pd.DataFrame(expanded_data)
    return df


def calculate_hand_bounding_box_iou(df):
    """
    Calculate the Intersection Over Union (IOU) for the bounding boxes of the left and right hands.
    """
    # Initialize IOU column with zeros
    df['HAND_BOUNDING_BOX_IOU'] = 0.0

    for i in range(len(df)):
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
            continue  # Skip if any hand's points are missing

        # Calculate bounding boxes
        left_hand_bbox = calculate_bbox(left_hand_points)
        right_hand_bbox = calculate_bbox(right_hand_points)

        # Calculate IOU
        df.at[i, 'HAND_BOUNDING_BOX_IOU'] = bbox_iou(left_hand_bbox, right_hand_bbox)

    return df['HAND_BOUNDING_BOX_IOU']

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
            feature_df[speed_feature_name] = calculate_speed(df, landmark, frame_rate)

        # 1. grip aperture (PGA) is the distance between the thumb and index finger
        # Calculate grip aperture for each frame and for each hand and add it to the feature DataFrame

        # 2. Wrist flexion / extension angle
        # calculated by the angle between wrist elbow joint and wrist middle finger mcps
        # Calculate wrist flexion/extension angle for each frame and each hand add it to the feature DataFrame

        # 3. compensation > iou between hands


        # 4. acceleration
        # 5. distance traversed
        #
        # ====================
        # 6. elbow flexion angle


        # 7. forearm pronation



        # 8. forearm supernation


        # 9. Palmer arch angle (angle between palm and the forearm)


        # 10. Shoulder flexion extension angle


        # 11. Wrist flexion extension angle

        # ====================
        # ====================

        # Object features
        # 12. speed of object
        # 13. trajectory deviation to the shortest path

        # This structure makes it easy to add more features to feature_df later
        feature_df['HAND_BOUNDING_BOX_IOU'] = calculate_hand_bounding_box_iou(df)

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

def write_features_to_db(df, db_util):
    """
    Writes the features from the DataFrame to the database, converting NaN values to NULL.
    """
    # Prepare the INSERT query template
    insert_query = """
    INSERT INTO VisionAnalysis.dbo.frame_features
    (id, frame_coordinate_id, frame_seq_number, recording_id, frame_rate, right_wrist_speed, right_elbow_speed, left_wrist_speed, left_elbow_speed, hand_bounding_box_iou)
    VALUES (NEWID(), ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """

    # Iterate over the DataFrame rows
    for index, row in df.iterrows():
        # Prepare the parameters for the current row, replacing NaN with None
        params = (
            row['frame_coordinate_id'],
            row['frame_seq_number'],
            row['recording_id'],
            row['frame_rate'],
            None if pd.isna(row['RIGHT_WRIST_SPEED']) else row['RIGHT_WRIST_SPEED'],
            None if pd.isna(row['RIGHT_ELBOW_SPEED']) else row['RIGHT_ELBOW_SPEED'],
            None if pd.isna(row['LEFT_WRIST_SPEED']) else row['LEFT_WRIST_SPEED'],
            None if pd.isna(row['LEFT_ELBOW_SPEED']) else row['LEFT_ELBOW_SPEED'],
            None if pd.isna(row['HAND_BOUNDING_BOX_IOU']) else row['HAND_BOUNDING_BOX_IOU']
        )

        # Execute the query with the current row's parameters
        db_util.insert_data(insert_query, params)


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
    print(df)

    df_with_features = calculate_features(df)

    write_features_to_db(df_with_features, db_util)

    # smoothed_df = smooth_data_moving_average(df_with_features, window_size=4)
    # visualize_features(smoothed_df)






