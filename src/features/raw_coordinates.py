import cv2
import mediapipe as mp
import argparse
import os
import json
import sys
import math
from pathlib import Path
detectron2_path = str((Path(__file__).parent.resolve() / "../../models/detectron2").absolute())

# Add the detectron2 installation path to sys.path
if detectron2_path not in sys.path:
    sys.path.append(detectron2_path)


from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog

from body_landmarks import BodyLandmark
sys.path.append('../')
from util import DatabaseUtil





mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
# interested_classes = ['person','cup','knife','fork','spoon']
interested_classes = ['cup','knife','fork','spoon']
global all_classes# 0 corresponds to person class in COCO dataset

# Function to initialize Detectron2's predictor with the specified model
def setup_detectron2(model_name="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", confidence_threshold=0.2):
    global all_classes
    cfg = get_cfg()
    #set cfg.MODEL.DEVICE to 'cpu'
    cfg.MODEL.DEVICE = 'cpu'
    cfg.merge_from_file(model_zoo.get_config_file(model_name))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)
    predictor = DefaultPredictor(cfg)
    all_classes = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
    return predictor

# Function to process a frame with Detectron2 and extract bounding boxes for specified classes
def detect_objects(frame, predictor, interested_classes):
    global all_classes  # Ensure all_classes is accessible
    outputs = predictor(frame)
    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes if instances.has("pred_boxes") else None
    classes = instances.pred_classes if instances.has("pred_classes") else None
    detected_objects = []

    # Initialize a dictionary to hold detected objects for each interested class
    detected_objects_dict = {interested_class: None for interested_class in interested_classes}

    if boxes is not None and classes is not None:
        for i, box in enumerate(boxes):
            class_index = classes[i].item()  # Get the class index as an integer
            class_name = all_classes[class_index]  # Use the class index to get the class name
            if class_name in interested_classes:
                box_tensor = box.data.numpy()  # Correctly access the tensor and convert to numpy
                # Instead of directly appending, assign the box to the correct class in the dictionary
                detected_objects_dict[class_name] = box_tensor.tolist()

    # Now, build the final list based on the order of interested_classes
    for class_name in interested_classes:
        if detected_objects_dict[class_name] is not None:
            detected_objects.append(detected_objects_dict[class_name])

    return detected_objects[:2]  # Return at most two detected objects


def serialize_objects(detected_objects):
    serialized_objects = []
    for obj in detected_objects:
        # Assuming obj format is [x1, y1, x2, y2]
        serialized_objects.append({
            'x1': round(obj[0], 4),
            'y1': round(obj[1], 4),
            'x2': round(obj[2], 4),
            'y2': round(obj[3], 4),
        })
    return serialized_objects


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
    try:
        left_shoulder = serialized_landmarks[BodyLandmark.LEFT_SHOULDER]
    except IndexError:
        left_shoulder = None

    try:
        right_shoulder = serialized_landmarks[BodyLandmark.RIGHT_SHOULDER]
    except IndexError:
        right_shoulder = None

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


def normalize_objects(detected_objects, left_shoulder, right_shoulder,frame_width, frame_height):
    if not left_shoulder or not right_shoulder:
        print("Shoulder landmarks are missing, cannot normalize objects.")
        return []

    # Directly calculate the reference distance (shoulder width) using 'x' and 'y'
    ref_distance = math.sqrt((left_shoulder.x - right_shoulder.x) ** 2 +
                             (left_shoulder.y - right_shoulder.y) ** 2)
    if ref_distance == 0:
        print("Reference distance is zero, cannot normalize objects.")
        return []

    normalized_objects = []
    for obj in detected_objects:
        obj_normalized = {
            'x1': obj['x1'] / frame_width,
            'y1': obj['y1'] / frame_height,
            'x2': obj['x2'] / frame_width,
            'y2': obj['y2'] / frame_height,
        }
        # Normalize each coordinate of the detected object based on the shoulder distance
        # and adjust the position relative to the left shoulder

        normalized_obj_corners = [
            {"x": round((obj_normalized['x1'] - left_shoulder.x) / ref_distance, 4),
             "y": round((obj_normalized['y1'] - left_shoulder.y) / ref_distance, 4)},
            {"x": round((obj_normalized['x2'] - left_shoulder.x) / ref_distance, 4),
             "y": round((obj_normalized['y2'] - left_shoulder.y) / ref_distance, 4)}
        ]

        normalized_objects.append(normalized_obj_corners)

    return normalized_objects

def drawMarkings(frame, left_shoulder, right_shoulder, detected_objects):
    for landmark in [left_shoulder, right_shoulder]:
        cv2.circle(frame, (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])), 5,
                   (0, 255, 0), -1)

    for obj in detected_objects:
        cv2.circle(frame, (int(obj[0]), int(obj[1])), 5, (255, 0, 0), -1)
        cv2.circle(frame, (int(obj[2]), int(obj[3])), 5, (255, 0, 0), -1)
        cv2.rectangle(frame, (int(obj[0]), int(obj[1])), (int(obj[2]), int(obj[3])), (255, 0, 0), 2)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return True  # Break the loop if '

def gendata(data_path, filename, desired_fps=10):

    all_pose_landmarks = []
    all_face_landmarks = []
    all_left_hand_landmarks = []
    all_right_hand_landmarks = []
    all_object_detections_1 = []
    all_object_detections_2 = []

    predictor = setup_detectron2()

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

        if data_path is None or filename is None:
            print("data_path or filename is None")
            return None

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

                left_hand_landmarks = results.right_hand_landmarks
                right_hand_landmarks = results.left_hand_landmarks

                # Collect landmarks without scaling
                pose_landmarks = serialize_landmarks(results.pose_landmarks, left_hand_landmarks, right_hand_landmarks)
                # if pose_landmarks is None: then appen none to all_pose_landmarks ,all_object_detections_1,all_object_detections_2 and continue
                if pose_landmarks is None or len(pose_landmarks) == 0:
                    all_pose_landmarks.append([None])
                    all_object_detections_1.append([None])
                    all_object_detections_2.append([None])
                    frame_idx += 1
                    continue
                # Call to normalize_landmarks
                pose_landmarks = normalize_landmarks(pose_landmarks)

                # Assuming you have left_shoulder and right_shoulder from serialized_landmarks
                left_shoulder = results.pose_landmarks.landmark[BodyLandmark.LEFT_SHOULDER.value]  # Adjust index as needed
                right_shoulder = results.pose_landmarks.landmark[BodyLandmark.RIGHT_SHOULDER.value]  # Adjust index as needed

                # Example usage after detecting and serializing objects
                detected_objects = detect_objects(frame, predictor, interested_classes)
                serialized_objects = serialize_objects(detected_objects)
                normalized_objects = normalize_objects(serialized_objects, left_shoulder, right_shoulder , frame.shape[1], frame.shape[0])


                if pose_landmarks:
                    all_pose_landmarks.append(pose_landmarks)
                else:
                    all_pose_landmarks.append(None)

                if len(normalized_objects) ==0:
                    all_object_detections_1.append([None])
                    all_object_detections_2.append([None])
                elif len(normalized_objects) == 1:
                    all_object_detections_1.append(normalized_objects[0])
                    all_object_detections_2.append([None])
                else:
                    all_object_detections_1.append(normalized_objects[0])
                    all_object_detections_2.append(normalized_objects[1])

                 # Display the current frame with landmarks DEBUG purpose only
                # if drawMarkings(frame, left_shoulder, right_shoulder, detected_objects):
                #     break

            frame_idx += 1

        cap.release()
        cv2.destroyAllWindows()
    return {
        'pose_landmarks': all_pose_landmarks,
        'object_detections_1': all_object_detections_1,
        'object_detections_2': all_object_detections_2
    }



def save_landmarks_to_db(recording_id, frame_rate, pose_landmarks, object_detections_1, object_detections_2):
    db_util = DatabaseUtil()  # Initialize your database utility class
    query = """
    INSERT INTO frame_coordinates (recording_id, frame_rate, coordinates, coordinates_object1, coordinates_object2)
    VALUES (?, ?, ?,?, ?)
    """
    params = (recording_id, frame_rate, json.dumps(pose_landmarks),json.dumps(object_detections_1),json.dumps(object_detections_2))
    inserted_id = db_util.insert_data(query, params, return_id=True)
    return inserted_id



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Data Processor with Custom Frame Rate for All Landmarks.')
    parser.add_argument('--data_path', default='../../data/raw/task_recordings',
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
        pose_landmarks=landmarks_data['pose_landmarks'],
        object_detections_1=landmarks_data['object_detections_1'],
        object_detections_2=landmarks_data['object_detections_2']
    )
    print(frame_coordinate_id)



