# -*- coding: utf-8 -*-
import pandas as pd
import subprocess
import argparse
import sys
import os
import re
from tqdm import tqdm
sys.path.append('../')
from util import DatabaseUtil


# Load the video recording info from an Excel file
def load_video_recording_info(file_path):
    with open(file_path, 'rb') as f:
        df = pd.read_excel(f)
    # df = pd.read_excel(file_path, sheet_name='video_recording_info',engine='openpyxl')

    return df



def update_session(db_util, session_type, session_number, user_id):
    # Check if the record already exists
    check_query = """
    SELECT id FROM VisionAnalysis.dbo.[session]
    WHERE session_type = ? AND session_number = ? AND user_id = ?;
    """
    params = (session_type, session_number, user_id)
    existing_records = db_util.fetch_data(check_query, params)

    if existing_records:
        # If record exists, return the existing session_id
        return existing_records[0][0]

    # If record does not exist, insert a new one
    insert_query = """
    INSERT INTO VisionAnalysis.dbo.[session] (session_type, session_number, user_id)
    OUTPUT INSERTED.id
    VALUES (?, ?, ?);
    """
    session_id = db_util.insert_data(insert_query, params, return_id=True)
    return session_id

# Update the recording table and return the recording_id
def update_recording(db_util, task, hand, date, time, flipped, session_id, file_name):
    query = """
    INSERT INTO VisionAnalysis.dbo.recording (task, hand, [date], [time], flipped, session_id, file_name)
    OUTPUT INSERTED.id
    VALUES (?,?, ?, ?, ?, ?, ?);
    """
    params = (task,hand, date, time, flipped, session_id, file_name)
    recording_id = db_util.insert_data(query, params, return_id=True)
    return recording_id

# Execute a Python script with arguments and return its output
def execute_script(script_name, args):
    result = subprocess.run(['python', script_name] + args, capture_output=True, text=True)
    if result.stdout:
        print(f"Output: {result.stdout}")
    if result.stderr:
        print(f"Error: {result.stderr}")
    # Use regular expression to find the first integer in the output
    match = re.search(r'\d+', result.stdout)
    if match:
        return int(match.group(0))  # Convert the found integer string to an int and return it
    else:
        return None

if __name__ == '__main__':
    excel_file_path = './video_recording_info.xlsx'
    db_util = DatabaseUtil()

    video_info_df = load_video_recording_info(excel_file_path)
    successful_rows = []  # Initialize the list to track successful rows

    for index, row in tqdm(video_info_df.iterrows(), total=video_info_df.shape[0]):

        if not (pd.isna(row['completed']) or row['completed'] == False):
            continue

        session_id = update_session(db_util, row['session_type'], row['assessment_id'], row['participant_id'])
        recording_id = update_recording(db_util, row['task_id'], row['hand'], row['date'], row['time'], False, session_id, row['file_name'])

        # Assume each step below is critical and must succeed for the row to be considered successfully processed
        try:
            # Step 3: Extract frame coordinates
            frame_coordinate_id = execute_script('raw_coordinates.py', ['--recording_id', str(recording_id), '--file_name', row['file_name']])
            # Step 4: Extract frame features
            execute_script('frame_features.py', ['--frame_coordinate_id', str(frame_coordinate_id)])
            # Step 5: Extract video features
            execute_script('video_features.py', ['--frame_coordinate_id', str(frame_coordinate_id)])
            video_info_df.at[index, 'recording_id'] = recording_id
            successful_rows.append(index)  # Track this row as successfully processed
        except Exception as e:
            print(f"Error processing row {index}: {e}")



    # Update the 'completed' column for successful rows
    for idx in successful_rows:
        video_info_df.at[idx, 'completed'] = True

    # Save the updated DataFrame back to the Excel file
    video_info_df.to_excel(excel_file_path, sheet_name='video_recording_info', index=False)
