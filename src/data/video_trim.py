from moviepy.editor import VideoFileClip
import os
import pandas as pd

def time_to_seconds(time_str):
    """Convert time format 'mm:ss:cs' to seconds."""
    hours,minutes, seconds, centiseconds = map(int, time_str.split(':'))
    return minutes * 60 + seconds + centiseconds / 100

def trim_video(input_file, start_time, end_time, output_file=None, output_folder='trimmed_videos'):
    # Ensure output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create the output file name if not provided
    if not output_file:
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = f"{base_name}_trimmed.mp4"

    # Full path for the output file
    output_path = os.path.join(output_folder, output_file)

    # Convert start and end time from 'min:sec' to seconds
    start_sec = time_to_seconds(start_time)
    end_sec = time_to_seconds(end_time)

    # Load the video file
    video = VideoFileClip(input_file)

    # Trim the video
    trimmed_video = video.subclip(start_sec, end_sec)

    # Write the trimmed video to a file
    trimmed_video.write_videofile(output_path, codec='libx264')

    # Close the video file to free resources
    video.close()

    return output_path

def main():
    meta_data_file_path = 'video_trim_timing_info.xlsx'
    sheet_name = 'task_timing_info'

    # Read data from Excel
    df = pd.read_excel(meta_data_file_path, sheet_name=sheet_name,skiprows=1,dtype=str)

    # Paths
    input_path = '../../data/raw/zoom_recordings'
    output_path = '../../data/raw/task_recordings'

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        participant_id = row['participant_id'] # Participant ID ['7001', '7002', '7003']
        assessment_id = row['assessment_id'] # [ 'E1','S1', 'P1','P2']
        task_id = row['task_id'] # Task ID ['DW':'Drinking Water', 'FPT':'Finger pointing task', 'OHA_L1','OHA_L2','OHA_L3' : 'Object hit ad avoid','FT' :'Finger Tracing']
        hand = row['hand'] # Hand ['left', 'right']
        start_time = row['start_time']
        end_time = row['end_time']
        video_file = row['video_file']
        saved_file_name = row['output_file']

        input_video = os.path.join(input_path, video_file)
        output_video_name = f"{participant_id}_{assessment_id}_{task_id}_{hand}.mp4"

        # Trim the video
        if pd.isna(saved_file_name) or saved_file_name.strip() == '':
            result_path = trim_video(input_video, start_time, end_time, output_video_name, output_path)
            print(f"Trimmed video saved to {result_path}")

if __name__ == "__main__":
    main()