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



def fetch_frame_features(ids):
    """
    Fetches frame features for the given IDs from the database.
    """
    placeholders = ', '.join(['?'] * len(ids))  # Create placeholders for parameterized query
    query = f"""
    SELECT frame_coordinate_id, frame_seq_number, recording_id, frame_rate, 
           right_elbow_speed, right_wrist_speed, left_elbow_speed, left_wrist_speed, hand_bounding_box_iou
    FROM VisionAnalysis.dbo.frame_features
    WHERE frame_coordinate_id IN ({placeholders})
    ORDER BY recording_id, frame_coordinate_id, frame_seq_number ASC;
    """
    db_util = DatabaseUtil()
    return db_util.fetch_data(query, ids)


def transform_data_to_dataframe(data):
    """
    Transforms the incoming data to a pandas DataFrame.
    """
    # Define column names based on the SELECT query
    columns = ['frame_coordinate_id', 'frame_seq_number', 'recording_id', 'frame_rate',
               'right_elbow_speed', 'right_wrist_speed', 'left_elbow_speed', 'left_wrist_speed',
               'hand_bounding_box_iou']

    # Convert the data to a DataFrame
    unpacked_data = [
        {col: val for col, val in zip(columns, row)} for row in data
    ]

    df = pd.DataFrame(unpacked_data)
    return df


def calculate_max_speed(group, column_name, window_size=4):
    """
    Calculate the maximum speed for a given column within a specified rolling window size.
    """
    # Apply a rolling window and calculate the max speed within the window for the specified column
    if column_name not in group.columns:
        raise ValueError(f"Column {column_name} not found in the DataFrame")
        # Calculate the rolling average within each window
    rolling_average = group[column_name].rolling(window=window_size, min_periods=1).mean()

    # Find the maximum value among the rolling averages
    max_average_speed = rolling_average.max()

    return round(max_average_speed, 4)


def calculate_features(df):
    """
    Calculate high-level features from the DataFrame.
    """
    # Group by recording_id, frame_coordinate_id to process each video separately
    grouped = df.groupby(['recording_id', 'frame_coordinate_id'])

    video_features = []

    for (recording_id, frame_coordinate_id), group in grouped:
        # Initialize a dictionary to store the features for the current group
        features = {
            'recording_id': recording_id,
            'frame_coordinate_id': frame_coordinate_id,
        }

        # Calculate each high-level feature and add it to the features dictionary
        features['right_max_wrist_speed'] = calculate_max_speed(group, 'right_wrist_speed')
        features['left_max_wrist_speed'] = calculate_max_speed(group, 'left_wrist_speed')

        # Example of adding another feature
        # features['another_feature'] = calculate_another_feature(group)

        # Append the calculated features for the current group to the list
        video_features.append(features)

    # Convert the list of high-level features to a DataFrame
    features_df = pd.DataFrame(video_features)
    return features_df


def save_features_to_database(features_df, db_util):
    """
    Saves the features DataFrame to the database.
    """
    # Define the SQL query template for insertion
    insert_query = """
    INSERT INTO VisionAnalysis.dbo.video_features
    (recording_id,frame_coordinate_id, right_max_wrist_speed, left_max_wrist_speed, right_wrist_time_to_peak_velocity, left_wrist_time_to_peak_velocity)
    VALUES (?,?, ?, ?, ?, ?);
    """

    # Iterate over the DataFrame rows
    for index, row in features_df.iterrows():
        # Prepare the data tuple for the current row
        # Note: Adjust the row accessors below based on the actual column names and calculations for time to peak velocity
        data_tuple = (
            row['recording_id'],
            row['frame_coordinate_id'],
            row['right_max_wrist_speed'],
            row['left_max_wrist_speed'],
            None,  # Placeholder for right_wrist_time_to_peak_velocity calculation
            None   # Placeholder for left_wrist_time_to_peak_velocity calculation
        )

        # Use the insert_data method of db_util to insert the data
        db_util.insert_data(insert_query, data_tuple)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract high-level frame features from frame coordinates')
    parser.add_argument('--frame_coordinate_id', nargs='+', type=int, default=[],
                        help='ID values of the frame_coordinate table entries to process')
    args = parser.parse_args()

    db_util = DatabaseUtil()

    # Fetch frame features data
    frame_feature_data = fetch_frame_features(args.frame_coordinate_id)

    # Transform data to DataFrame
    df = transform_data_to_dataframe(frame_feature_data)

    features_df = calculate_features(df)
    save_features_to_database(features_df, db_util)
    print(features_df)







