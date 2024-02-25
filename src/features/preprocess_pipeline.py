# # read the data from video_recording_info excel file. load data in the sheet name 'video_recording_info' and store it in a dataframe.
# # column names: user_id,	session_type,	session_number,	task,	date,	time,	file_name,	completed
# # filter all the rows which aare ahveing 'completed' status as FASLE or null
# # iterate through each row and get the video file path and the session number
#
# run steps 1 to 5 for each of the raw
#
# # step 1: update session table with the new data
#
# db table structure
# INSERT INTO VisionAnalysis.dbo.[session]
# (session_type, session_number, user_id)
# VALUES('', 0, 0);
# get the id of the [session] table entry which was just inserted and keep it (session_id)
#
#
#
# # step 2: update recording table with the new data
# INSERT INTO VisionAnalysis.dbo.recording
# (task, [date], [time], flipped, session_id, file_name)
# VALUES('', '', '', 0, 0, '');
#
# get the the details from the excel sheet dataframe and the session_id froom step 1:
# set flipped to FALSE
# get the id of the [recording] table entry which was just inserted and keep it as recording_id
#
#
# # setp 3: extract frame cooordinates > execute raw_coordinates.py
# pass recording_id and video file_name as arguments . extract argument values from above data frames and saved values. pass fps as 10
# get the return value aand savee it as frame_coordinate_id
#
# #sttep 4: extract frame features  > execute frame_features.py
# pass frame_coordinate_id,recording_id  as argument.
#
#
# # step 5: extract video features > execute video_features.py
# pass frame_coordinate_id,recording_id  as argument.

import pandas as pd
import subprocess
import argparse
import sys
import os
import re
sys.path.append('../')
from util import DatabaseUtil

# Load the video recording info from an Excel file
def load_video_recording_info(file_path):
    df = pd.read_excel(file_path, sheet_name='video_recording_info')

    return df

# Update the session table and return the session_id
def update_session(db_util, session_type, session_number, user_id):
    query = """
    INSERT INTO VisionAnalysis.dbo.[session] (session_type, session_number, user_id)
    OUTPUT INSERTED.id
    VALUES (?, ?, ?);
    """
    params = (session_type, session_number, user_id)
    session_id = db_util.insert_data(query, params, return_id=True)
    return session_id

# Update the recording table and return the recording_id
def update_recording(db_util, task, date, time, flipped, session_id, file_name):
    query = """
    INSERT INTO VisionAnalysis.dbo.recording (task, [date], [time], flipped, session_id, file_name)
    OUTPUT INSERTED.id
    VALUES (?, ?, ?, ?, ?, ?);
    """
    params = (task, date, time, flipped, session_id, file_name)
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

    for index, row in video_info_df.iterrows():

        if not (pd.isna(row['completed']) or row['completed'] == False):
            continue

        session_id = update_session(db_util, row['session_type'], row['session_number'], row['user_id'])
        recording_id = update_recording(db_util, row['task'], row['date'], row['time'], False, session_id, row['file_name'])

        # Assume each step below is critical and must succeed for the row to be considered successfully processed
        try:
            # Step 3: Extract frame coordinates
            frame_coordinate_id = execute_script('raw_coordinates.py', ['--recording_id', str(recording_id), '--file_name', row['file_name']])
            # Step 4: Extract frame features
            execute_script('frame_features.py', ['--frame_coordinate_id', str(frame_coordinate_id)])
            # Step 5: Extract video features
            execute_script('video_features.py', ['--frame_coordinate_id', str(frame_coordinate_id)])
            successful_rows.append(index)  # Track this row as successfully processed
        except Exception as e:
            print(f"Error processing row {index}: {e}")

    # Update the 'completed' column for successful rows
    for idx in successful_rows:
        video_info_df.at[idx, 'completed'] = True

    # Save the updated DataFrame back to the Excel file
    video_info_df.to_excel(excel_file_path, sheet_name='video_recording_info', index=False)
