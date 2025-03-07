import cv2
import mediapipe as mp
import logging
import os

def blur_faces_in_image(input_image_path, padding=0.2):
    """Blurs faces in an image."""
    image = cv2.imread(input_image_path)

    # Initialize mediapipe's face detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces
    results = face_detection.process(image_rgb)

    # If faces are detected
    if results.detections:
        for detection in results.detections:
            # Get bounding box coordinates
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih * 1.15), int(bboxC.width * iw), int(bboxC.height * ih * 0.6)

            # Adjust the bounding box to cover only the eye region (40% of the face height)
            eye_region_height = int(h * 0.4)
            x1 = max(0, int(x - padding * w))
            y1 = max(0, int(y - padding * eye_region_height))
            x2 = min(iw, int(x + w + padding * w))
            y2 = min(ih, int(y + eye_region_height + padding * eye_region_height))

            # Extract and blur the eye region
            eye_area = image[y1:y2, x1:x2]
            eye_area = cv2.GaussianBlur(eye_area, (21, 99), 30)
            image[y1:y2, x1:x2] = eye_area

    # Create output file path
    output_image_path = generate_output_path(input_image_path)

    # Save the processed image
    cv2.imwrite(output_image_path, image)
    logging.info(f"Processed image saved at {output_image_path}")


def blur_faces_in_video(input_video_path, padding=0.2):
    """Blurs faces in a video."""
    cap = cv2.VideoCapture(input_video_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create output file path
    output_video_path = generate_output_path(input_video_path)

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Initialize mediapipe's face detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        results = face_detection.process(frame_rgb)

        # If faces are detected
        if results.detections:
            for detection in results.detections:
                # Get bounding box coordinates
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih * 1.15), int(bboxC.width * iw), int(bboxC.height * ih * 0.5)

                # Adjust the bounding box to cover only the eye region (40% of the face height)
                eye_region_height = int(h * 0.4)
                x1 = max(0, int(x - padding * w))
                y1 = max(0, int(y - padding * eye_region_height))
                x2 = min(iw, int(x + w + padding * w))
                y2 = min(ih, int(y + eye_region_height + padding * eye_region_height))

                # Extract and blur the eye region
                eye_area = frame[y1:y2, x1:x2]
                eye_area = cv2.GaussianBlur(eye_area, (21, 99), 30)
                frame[y1:y2, x1:x2] = eye_area

        # Write the frame to the output video
        out.write(frame)

    cap.release()
    out.release()
    logging.info(f"Processed video saved at {output_video_path}")


def generate_output_path(input_path):
    """Generates the output file path by appending '_blurred' to the original file name."""
    dir_name, file_name = os.path.split(input_path)
    base_name, ext = os.path.splitext(file_name)
    output_file_name = f"{base_name}_blurred{ext}"
    return os.path.join(dir_name, output_file_name)


def main():
    """Main method to call image and video blurring functions."""
    # List of image and video paths
    image_paths = [
        "data/faceimages/grasp_1_full.png",
        "data/faceimages/grasp_2_full.png"
    ]

    video_paths = [
        "data/faceimages/7002_S1_DW_right.mp4",
        "data/faceimages/7002_S1_DW_left.mp4"
    ]

    # Process images
    # for image_path in image_paths:
    #     blur_faces_in_image(image_path, padding=0.3)

    # Process videos
    for video_path in video_paths:
        blur_faces_in_video(video_path, padding=0.3)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
