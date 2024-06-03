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
from util import read_feature_file
from util.feature_util import *


def smooth_data(data, window_length=5, polyorder=2):
    """
    Smooths the data using the Savitzky-Golay filter to remove sudden spikes.
    Ensures the window length is odd and not greater than the length of the data.
    """
    # Ensure window length is odd and not greater than the length of data
    if window_length > len(data) or window_length % 2 == 0:
        window_length = min(len(data), window_length)
        if window_length % 2 == 0:
            window_length += 1  # Make sure it's odd

    # Smooth the values using Savitzky-Golay filter
    if len(data) >= window_length:
        return savgol_filter(data, window_length, polyorder)
    else:
        return data  # Fallback to original if not enough data

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


def calculate_velocity_peaks(group, column_name, percentile=95, window_length=5, polyorder=2):
    """
    Calculate the number of velocity peaks for a given velocity column in uppercase,
    excluding None and NaN values, and smoothing the values to remove sudden spikes before calculating the percentile.
    """
    # Ensure the column name is in uppercase to match the DataFrame
    column_name = column_name.upper()

    if column_name not in group.columns:
        raise ValueError(f"Column {column_name} not found in the DataFrame")

    # Exclude None and NaN values
    valid_values = group[column_name].dropna()

    # Ensure window length is odd and not greater than the length of valid_values
    if window_length > len(valid_values) or window_length % 2 == 0:
        window_length = min(len(valid_values), window_length)
        if window_length % 2 == 0:
            window_length += 1  # Make sure it's odd

    # Smooth the values using Savitzky-Golay filter to remove sudden spikes
    # The filter might not be applied if valid_values is too short, check the length first
    if len(valid_values) >= window_length:
        smoothed_values = savgol_filter(valid_values, window_length, polyorder)
    else:
        smoothed_values = valid_values  # Fallback to original if not enough data

    # Calculate the 80th percentile for the smoothed velocity column
    threshold = np.percentile(smoothed_values, percentile)

    # Identify regions where smoothed velocity is above the threshold
    above_threshold = smoothed_values > threshold

    # Since we filtered NaNs from valid_values and potentially applied smoothing, we need to align indices
    # Create a series for above_threshold with group's index
    above_threshold_series = pd.Series(above_threshold, index=valid_values.index)

    # Count transitions from below to above threshold as the start of a new peak
    peak_starts = (above_threshold_series.shift(1, fill_value=above_threshold_series.iloc[
        0]) < above_threshold_series).sum()

    return peak_starts


def calculate_direction_changes(group, column_name):
    """
    Calculates the number of direction changes for a given speed column.
    Smoothes the speed, calculates the acceleration, then the jerk, and finally counts the zero crossings.
    """
    column_name = column_name.upper()
    if column_name not in group.columns:
        raise ValueError(f"Column {column_name} not found in the DataFrame")

    valid_speeds = group[column_name].dropna()
    smoothed_speeds = smooth_data(valid_speeds)
    acceleration = np.gradient(smoothed_speeds)
    # jerk = np.gradient(acceleration)
    sign_changes = np.diff(np.sign(acceleration))
    zero_crossings = np.where(sign_changes != 0)[0]
    number_of_direction_changes = len(zero_crossings)

    return number_of_direction_changes

def calculate_direction_changes_rate(group, column_name, frame_rate):
    """
    Calculates the rate of direction changes per second for a given speed column,
    using the number of direction changes calculated previously and the video's frame rate.
    """
    number_of_direction_changes = calculate_direction_changes(group, column_name)
    rate_of_direction_changes = number_of_direction_changes / (len(group) / frame_rate)
    return rate_of_direction_changes

def calculate_completion_time(group, frame_rate):
    """
    Calculates the completion time of an action using the number of frames
    and the frame rate.
    """
    number_of_frames = len(group)
    completion_time = number_of_frames / frame_rate
    return completion_time


def calculate_speed_metrics(group, column_name, frame_rate):
    """
    Calculates various speed metrics including max speed, avg speed, standard deviation,
    quartiles (Q1, Q2, Q3), and time to peak velocity for a given wrist speed column.
    """
    # Ensure the column name is in uppercase to match the DataFrame
    column_name = column_name.upper()

    if column_name not in group.columns:
        raise ValueError(f"Column {column_name} not found in the DataFrame")

    # Smooth the speed data
    valid_speeds = group[column_name].dropna()
    smoothed_speeds = smooth_data(valid_speeds)

    # Calculate metrics
    max_speed = round(smoothed_speeds.max(),4)
    avg_speed = round(smoothed_speeds.mean(),4)
    std_speed = round(smoothed_speeds.std(),4)
    q1_speed = round(np.percentile(smoothed_speeds, 25),4)
    q2_speed = round(np.percentile(smoothed_speeds, 50),4)
    q3_speed = round(np.percentile(smoothed_speeds, 75),4)

    # Calculate time to peak velocity
    # Find the index of the max speed, divide by frame rate to convert to time
    peak_velocity_index = round(np.argmax(smoothed_speeds),4)
    time_to_peak_velocity = round(peak_velocity_index / frame_rate,4)

    return max_speed, avg_speed, std_speed, q1_speed, q2_speed, q3_speed, time_to_peak_velocity


def calculate_total_traversed_distance(group, column_name, frame_rate=10):
    """
    Calculates the total traversed distance for a given wrist speed column.
    NA values are replaced with the nearest non-null value. Speeds are then smoothed,
    and the total distance is calculated by integrating the speed over time.
    """
    column_name = column_name.upper()

    if column_name not in group.columns:
        raise ValueError(f"Column {column_name} not found in the DataFrame")

    # Replace NaN values with the nearest non-null value
    valid_speeds = group[column_name].fillna(method='ffill').fillna(method='bfill')

    # Smooth the speeds using Savitzky-Golay filter with window size 4
    # Ensure window size is appropriate for the length of data
    window_length = min(4, len(valid_speeds))
    if window_length % 2 == 0:
        window_length += 1  # Ensure window length is odd

    if len(valid_speeds) > 1:
        smoothed_speeds = savgol_filter(valid_speeds, window_length, 2)
    else:
        smoothed_speeds = valid_speeds

    # Calculate the distance for each frame and sum to get the total distance
    distances_per_frame = smoothed_speeds / frame_rate
    total_distance = distances_per_frame.sum()

    return total_distance


def calculate_total_trajectory_error(group, column_name):
    """
    Calculates the total trajectory error for a given column.
    NA values are replaced with the nearest non-null value before accumulating the total value.
    """


    if column_name not in group.columns:
        raise ValueError(f"Column {column_name} not found in the DataFrame")

    # Replace NaN values with the nearest non-null value using forward-fill and back-fill
    valid_values = group[column_name].fillna(method='ffill').fillna(method='bfill')

    # Accumulate the total trajectory error
    total_trajectory_error = valid_values.sum()

    return total_trajectory_error

def calculate_features(df):
    """
    Calculate high-level features from the DataFrame.
    """
    # Group by recording_id, frame_coordinate_id to process each video separately
    # check df is not empty
    if df is None or df.empty:
        return pd.DataFrame()

    grouped = df.groupby(['recording_id', 'frame_coordinate_id'])

    default_frame_rate = 10
    frame_rate = df['FRAME_RATE'][0] if 'FRAME_RATE' in df.columns and not df['FRAME_RATE'].isna().all() else default_frame_rate

    video_features = []

    for (recording_id, frame_coordinate_id), group in grouped:
        # Initialize a dictionary to store the features for the current group
        features = {
            'recording_id': recording_id,
            'frame_coordinate_id': frame_coordinate_id,
        }

        # Calculate each high-level feature and add it to the features dictionary
        features['RIGHT_WRIST_SPEED_MAX'] = calculate_max_speed(group, 'RIGHT_WRIST_SPEED')
        features['LEFT_WRIST_SPEED_MAX'] = calculate_max_speed(group, 'LEFT_WRIST_SPEED')

        # 1. number of velocity peaks
        #  get the RIGHT_WRIST_SPEED  colums get the velocity distribution over the time. and get the 80% percentile and identify the discrete number of regions where the velocity is above the 80% percentile.
        #  these are know as the velocity peaks.
        #  calculate for both right and left wrist and add to features dictionary. 'RIGHT_WRIST_numnber_of_velocity_peaks' and 'LEFT_WRIST_numnber_of_velocity_peaks' and 'RIGHT_WRIST_numnber_of_velocity_peaks'
        features['RIGHT_WRIST_NUMBER_OF_VELOCITY_PEAKS'] = calculate_velocity_peaks(group, 'RIGHT_WRIST_SPEED')
        features['LEFT_WRIST_NUMBER_OF_VELOCITY_PEAKS'] = calculate_velocity_peaks(group, 'LEFT_WRIST_SPEED')


        # 2. number of direction changes (Number of acceleration zero crossings )
        #  get the RIGHT_WRIST_SPEED and first smooth it as in the prevoiouse step. then calculate the acceleration. then get the rate of acceleratioon. (JERK)  then get the accellaton zero crossings. add to features dictionary. and 'LEFT_WRIST_numnber_of_direction_changes' and 'RIGHT_WRIST_numnber_of_direction_changes'
        # do for both right and left wrist
        features['RIGHT_WRIST_NUMBER_OF_DIRECTION_CHANGES'] = calculate_direction_changes(group, 'RIGHT_WRIST_SPEED')
        features['LEFT_WRIST_NUMBER_OF_DIRECTION_CHANGES'] = calculate_direction_changes(group, 'LEFT_WRIST_SPEED')

        # 3. ratet of direction changes (per second)
        # get the RIGHT_WRIST_JERK column and get the zero crossings. divide by the total time of the video. add to features dictionary. 'RIGHT_WRIST_rate_of_direction_changes' and 'LEFT_WRIST_rate_of_direction_changes' and 'RIGHT_WRIST_rate_of_direction_changes'
        features['RIGHT_WRIST_RATE_OF_DIRECTION_CHANGES'] = calculate_direction_changes_rate(group, 'RIGHT_WRIST_SPEED', frame_rate)
        features['LEFT_WRIST_RATE_OF_DIRECTION_CHANGES'] = calculate_direction_changes_rate(group, 'LEFT_WRIST_SPEED', frame_rate)

        # 4. completion time
        # get the completion time by refering to the number of frames (number of rows) and the frame rate
        features['COMPLETION_TIME'] = calculate_completion_time(group, frame_rate)

        # 5. peak speed/ AVG speed/ std/ q1,q2,q3/ time to peak velocity
        # get the RIGHT_WRIST_SPEED column and get the max speed, avg speed, std, q1,q2,q3, time to peak velocity measaures and update the features dictionary.
        #  do for both right and left wrist
        # Calculate speed metrics for right and left wrists
        metrics_right = calculate_speed_metrics(group, 'RIGHT_WRIST_SPEED', frame_rate)
        metrics_left = calculate_speed_metrics(group, 'LEFT_WRIST_SPEED', frame_rate)

        # Unpack metrics into features dictionary
        (features['RIGHT_WRIST_MAX_SPEED'], features['RIGHT_WRIST_AVG_SPEED'], features['RIGHT_WRIST_STD_SPEED'],
         features['RIGHT_WRIST_Q1_SPEED'], features['RIGHT_WRIST_Q2_SPEED'], features['RIGHT_WRIST_Q3_SPEED'],
         features['RIGHT_WRIST_TIME_TO_PEAK_VELOCITY']) = metrics_right

        (features['LEFT_WRIST_MAX_SPEED'], features['LEFT_WRIST_AVG_SPEED'], features['LEFT_WRIST_STD_SPEED'],
         features['LEFT_WRIST_Q1_SPEED'], features['LEFT_WRIST_Q2_SPEED'], features['LEFT_WRIST_Q3_SPEED'],
         features['LEFT_WRIST_TIME_TO_PEAK_VELOCITY']) = metrics_left


        #  6. total traversed distance
        # refer to the 'RIGHT_WRIST_SPEED' column aand referring too the frame rate get the distaance. get the toal traveresed idstnaace over the time.
        # add to the feature dictionary as 'RIGHT_WRIST_TOTAL_TRAVERSED_DISTANCE'
        features['RIGHT_WRIST_TOTAL_TRAVERSED_DISTANCE'] = calculate_total_traversed_distance(group,'RIGHT_WRIST_SPEED',frame_rate)
        features['LEFT_WRIST_TOTAL_TRAVERSED_DISTANCE'] = calculate_total_traversed_distance(group,'LEFT_WRIST_SPEED',frame_rate)

        # 7. total trajectory error
        #  refer to the group[object_1_trajectory_deviation] and accumilatet over the givven period of time. add to the feature dictionary aas 'TOTAL_TRAJECTORY_ERROR'
        features['TOTAL_TRAJECTORY_ERROR'] = calculate_total_trajectory_error(group, 'object_1_trajectory_deviation')


        video_features.append(features)

    # Convert the list of high-level features to a DataFrame
    features_df = pd.DataFrame(video_features)
    return features_df


def save_features_to_database(features_df, db_util):
    """
    Saves the features DataFrame to the database.
    """

    insert_query = """
        INSERT INTO VisionAnalysis.dbo.video_features
        (frame_coordinate_id, recording_id, right_max_wrist_speed, left_max_wrist_speed, 
        right_wrist_time_to_peak_velocity, left_wrist_time_to_peak_velocity, 
        right_wrist_number_of_velocity_peaks, left_wrist_number_of_velocity_peaks, 
        right_wrist_number_of_direction_changes, left_wrist_number_of_direction_changes, 
        right_wrist_rate_of_direction_changes, left_wrist_rate_of_direction_changes, 
        completion_time, right_wrist_max_speed, right_wrist_avg_speed, 
        right_wrist_std_speed, right_wrist_q1_speed, right_wrist_q2_speed, 
        right_wrist_q3_speed, left_wrist_max_speed, left_wrist_avg_speed, 
        left_wrist_std_speed, left_wrist_q1_speed, left_wrist_q2_speed, 
        left_wrist_q3_speed, right_wrist_total_traversed_distance, 
        left_wrist_total_traversed_distance, total_trajectory_error)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);
        """

    # Iterate over the DataFrame rows
    for index, row in features_df.iterrows():
        data_tuple = (
            row['frame_coordinate_id'],
            row['recording_id'],
            row['RIGHT_WRIST_SPEED_MAX'],
            row['LEFT_WRIST_SPEED_MAX'],
            row['RIGHT_WRIST_TIME_TO_PEAK_VELOCITY'], row['LEFT_WRIST_TIME_TO_PEAK_VELOCITY'],
            row['RIGHT_WRIST_NUMBER_OF_VELOCITY_PEAKS'], row['LEFT_WRIST_NUMBER_OF_VELOCITY_PEAKS'],
            row['RIGHT_WRIST_NUMBER_OF_DIRECTION_CHANGES'], row['LEFT_WRIST_NUMBER_OF_DIRECTION_CHANGES'],
            row['RIGHT_WRIST_RATE_OF_DIRECTION_CHANGES'], row['LEFT_WRIST_RATE_OF_DIRECTION_CHANGES'],
            row['COMPLETION_TIME'], row['RIGHT_WRIST_MAX_SPEED'], row['RIGHT_WRIST_AVG_SPEED'],
            row['RIGHT_WRIST_STD_SPEED'], row['RIGHT_WRIST_Q1_SPEED'], row['RIGHT_WRIST_Q2_SPEED'],
            row['RIGHT_WRIST_Q3_SPEED'], row['LEFT_WRIST_MAX_SPEED'], row['LEFT_WRIST_AVG_SPEED'],
            row['LEFT_WRIST_STD_SPEED'], row['LEFT_WRIST_Q1_SPEED'], row['LEFT_WRIST_Q2_SPEED'],
            row['LEFT_WRIST_Q3_SPEED'], row['RIGHT_WRIST_TOTAL_TRAVERSED_DISTANCE'],
            row['LEFT_WRIST_TOTAL_TRAVERSED_DISTANCE'], row['TOTAL_TRAJECTORY_ERROR']
        )

        # Use the insert_data method of db_util to insert the data
        db_util.insert_data(insert_query, data_tuple)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract high-level frame features from frame coordinates')
    parser.add_argument('--frame_coordinate_id', nargs='+', type=int, default='',
                        help='ID values of the frame_coordinate table entries to process')
    args = parser.parse_args()

    db_util = DatabaseUtil()

    # Fetch frame features data
    frame_feature_data = read_feature_file(args.frame_coordinate_id)

    # Transform data to DataFrame
    # df = transform_data_to_dataframe(frame_feature_data)
    # frame_feature_data.columns = ['frame_coordinate_id', 'frame_seq_number', 'recording_id', 'frame_rate',
    features_df = calculate_features(frame_feature_data)
    if not features_df.empty:
        save_features_to_database(features_df, db_util)
    else:
        print('No features extracted')







