import cv2
import mediapipe as mp
import logging

def blur_face_in_image(input_image_path, output_image_path, padding=0.2):
    # Load the image
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
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            # Increase the bounding box by padding factor
            x1 = max(0, int(x - padding * w))
            y1 = max(0, int(y))
            x2 = min(iw, int(x + w + padding * w))
            y2 = min(ih, int(y + h/3 + padding * h))

            # Extract the face area with padding
            face = image[y1:y2, x1:x2]

            # Apply a Gaussian blur to the face area
            face = cv2.GaussianBlur(face, (99, 99), 30)

            # Replace the original face area with the blurred face
            image[y1:y2, x1:x2] = face

    # Save the new image
    cv2.imwrite(output_image_path, image)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log_error(function_name, message):
    logging.error(f"Error in {function_name}: {message}")
#
# # Example usage

# output_image_path = "P7001_affected_annoccddtated.png"
# output_image_path = "P7001_unaffected_annotated.png"
# output_image_path = "P7002_affected_annotated.png"
# output_image_path = "P7002_unaffected_annotated.png"

input_image_path = "data/faceimages/grasp_8_full.png"  # Replace with your image path
output_image_path = "data/faceimages/grasp_8_full_blurred.png"  # Replace with the desired output path
blur_face_in_image(input_image_path, output_image_path, padding=0.1)
