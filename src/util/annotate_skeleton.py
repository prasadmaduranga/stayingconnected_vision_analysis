import cv2
import mediapipe as mp

# Initialize MediaPipe's hand detector
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load the image
 # Path to your input image
# image_path = "P7001_affected.png"
# image_path = "P7001_unaffected.png"
# image_path = "P7002_affected.png"
image_path = "data/backup_images/P7002_unaffected.png"

image = cv2.imread(image_path)

# Convert the BGR image to RGB for MediaPipe processing
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Initialize the Hands model from MediaPipe
with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
    # Process the image and detect hand landmarks
    results = hands.process(image_rgb)

    # Check if landmarks were detected
    if results.multi_hand_landmarks:
        # Iterate through the detected hands and landmarks
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks and connections on the image
            mp_drawing.draw_landmarks(
                image,  # The original image
                hand_landmarks,  # The landmarks detected
                mp_hands.HAND_CONNECTIONS,  # The connections between landmarks
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=2),  # Landmark style
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1)  # Connection style
            )

# Save the annotated image
output_image_path = "P7001_affected.jpg"

# output_image_path = "P7001_affected_annotated.png"
# output_image_path = "P7001_unaffected_annotated.png"
# output_image_path = "P7002_affected_annotated.png"
output_image_path = "data/backup_images/P7002_unaffected_annotated.png"
cv2.imwrite(output_image_path, image)

# Optionally, display the result
# cv2.imshow("Annotated Image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

print(f"Annotated image saved as {output_image_path}")
