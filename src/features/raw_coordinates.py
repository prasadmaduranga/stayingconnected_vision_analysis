import cv2
import mediapipe as mp
import argparse
import os
import json
import sys
import math
from body_landmarks import BodyLandmark
sys.path.append('../')
from util import DatabaseUtil


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

def serialize_landmarks(pose_landmarks=None, left_hand_landmarks=None, right_hand_landmarks=None):
    """Serializes landmarks without scaling by image dimensions."""
    if not pose_landmarks:
        return []

    serialized = [None] * len(BodyLandmark)

    def serialize_individual_landmarks(landmarks, offset,limit=None):
        if not landmarks:
            return
        for i, landmark in enumerate(landmarks.landmark):
            if limit is not None and i > limit:
                break
            serialized[i + offset] = {
                'x': round(landmark.x, 4),
                'y': round(landmark.y, 4),
                'z': round(landmark.z, 4)
            }

    pose_offset = BodyLandmark.NOSE  # Assuming this is the first pose landmark
    left_hand_offset = BodyLandmark.LEFT_WRIST  # Adjust based on enum
    right_hand_offset = BodyLandmark.RIGHT_WRIST  # Adjust based on enum

    serialize_individual_landmarks(pose_landmarks, pose_offset,limit=BodyLandmark.RIGHT_ELBOW)
    serialize_individual_landmarks(left_hand_landmarks, left_hand_offset)
    serialize_individual_landmarks(right_hand_landmarks, right_hand_offset)

    return serialized


def calculate_distance(landmark1, landmark2):
    """Calculate the Euclidean distance between two landmarks and round to 4 decimal points."""
    distance = math.sqrt((landmark1['x'] - landmark2['x']) ** 2 +
                         (landmark1['y'] - landmark2['y']) ** 2 +
                         (landmark1['z'] - landmark2['z']) ** 2)
    return round(distance, 4)

def normalize_landmarks(serialized_landmarks):
    """Normalize landmarks relative to the distance between the shoulders."""
    left_shoulder = serialized_landmarks[BodyLandmark.LEFT_SHOULDER]
    right_shoulder = serialized_landmarks[BodyLandmark.RIGHT_SHOULDER]

    # Ensure both shoulders are present
    if left_shoulder is None or right_shoulder is None:
        print("Shoulder landmarks are missing, cannot normalize.")
        return [None] * len(BodyLandmark)

    # Calculate the reference distance (shoulder width)
    ref_distance = calculate_distance(left_shoulder, right_shoulder)
    if ref_distance == 0:
        print("Reference distance is zero, cannot normalize.")
        return [None] * len(BodyLandmark)

    # Normalize landmarks
    normalized_landmarks = []
    for landmark in serialized_landmarks:
        if landmark is not None:
            normalized_landmarks.append({
                'x': round((landmark['x'] - left_shoulder['x']) / ref_distance, 4),
                'y': round((landmark['y'] - left_shoulder['y']) / ref_distance, 4),
                'z': round((landmark['z'] - left_shoulder['z']) / ref_distance, 4)
            })
        else:
            normalized_landmarks.append(None)

    return normalized_landmarks

def gendata(data_path, filename, desired_fps=10):
    all_pose_landmarks = []
    all_face_landmarks = []
    all_left_hand_landmarks = []
    all_right_hand_landmarks = []

    with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            enable_segmentation=True,
            refine_face_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.8) as holistic:

        # Construct the full file path from the data_path and filename
        # handle file path errors
        file_path = os.path.join(data_path, filename)
        if not os.path.exists(file_path):
            print(f"File does not exist: {file_path}")
            return None



        # Open the video file
        cap = cv2.VideoCapture(file_path)

        if not cap.isOpened():
            print(f"Failed to open video file: {filename}")
            return None  # Return None or appropriate error handling

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps / desired_fps)

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # Collect landmarks without scaling
                pose_landmarks = serialize_landmarks(results.pose_landmarks, results.left_hand_landmarks, results.right_hand_landmarks)
                # Call to normalize_landmarks
                pose_landmarks = normalize_landmarks(pose_landmarks)
                if pose_landmarks:
                    all_pose_landmarks.append(pose_landmarks)
                else:
                    all_pose_landmarks.append(None)

            frame_idx += 1

        cap.release()

    return {
        'pose_landmarks': all_pose_landmarks
    }



def save_landmarks_to_db(recording_id, frame_rate, pose_landmarks):
    db_util = DatabaseUtil()  # Initialize your database utility class
    query = """
    INSERT INTO frame_coordinates (recording_id, frame_rate, coordinates)
    VALUES (?, ?, ?)
    """
    params = (recording_id, frame_rate, json.dumps(pose_landmarks))
    inserted_id = db_util.insert_data(query, params, return_id=True)
    return inserted_id



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Data Processor with Custom Frame Rate for All Landmarks.')
    parser.add_argument('--data_path', default='../../data/raw/test2',
                        help='Path to the directory containing video files.')
    parser.add_argument('--file_name')
    parser.add_argument('--fps', default=10)
    parser.add_argument('--recording_id')
    args = parser.parse_args()

    landmarks_data = gendata(args.data_path, args.file_name)

    if landmarks_data is None:
        print("No landmarks data was generated.")
        sys.exit(1)

    recording_id = args.recording_id
    frame_rate = args.fps
    frame_coordinate_id = None

    # Now call save_landmarks_to_db with the collected landmarks data
    frame_coordinate_id = save_landmarks_to_db(
        recording_id=recording_id,
        frame_rate=frame_rate,
        pose_landmarks=landmarks_data['pose_landmarks']
    )
    print(frame_coordinate_id)



