import cv2
import dlib
from imutils import face_utils
from scipy.spatial import distance
import numpy as np

# Initialize video capture and models
video_capture = cv2.VideoCapture(0)  # Use the default camera
face_model = dlib.get_frontal_face_detector()
landmark_model = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Thresholds for yawn detection
yawn_min_thresh = 7
yawn_max_thresh = 20

# Drowsiness detection thresholds
EYE_ASPECT_RATIO_THRESHOLD = 0.25
EYE_ASPECT_RATIO_CONSEC_FRAMES = 40
COUNTER = 0
drowsiness_detected = False
drowsiness_count = 0

# Extract indexes of facial landmarks for the left and right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

# Function to calculate eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2 * C)
    return ear

# Function to calculate mouth aspect ratio (MAR)
def mouth_aspect_ratio(top_lip_point, bottom_lip_point, left_corner, right_corner):
    vertical_distance = bottom_lip_point[1] - top_lip_point[1]
    horizontal_distance = right_corner[0] - left_corner[0]
    mar = vertical_distance / horizontal_distance
    return mar

# Variables for initial calibration
initial_vertical_distance = None
initial_horizontal_distance = None
is_initial_distance_set = False

# Variables for yawning detection
yawn_detected = False
yawn_count = 0
mouth_aspect_ratios = []  # List to store MAR over frames for smoothing

# Main loop for video processing
while True:
    success, frame = video_capture.read()
    if not success:
        break

    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_model(img_gray)

    for face in faces:
        shapes = landmark_model(img_gray, face)
        shape = face_utils.shape_to_np(shapes)

        # Yawn Detection (MAR)
        top_lip_point = shape[51]  # Top midpoint of the upper lip
        bottom_lip_point = shape[57]  # Bottom midpoint of the lower lip
        left_corner = shape[48]
        right_corner = shape[54]

        mar = mouth_aspect_ratio(top_lip_point, bottom_lip_point, left_corner, right_corner)
        mouth_aspect_ratios.append(mar)

        if len(mouth_aspect_ratios) > 10:  # Only keep the last 10 values for smoothing
            mouth_aspect_ratios.pop(0)
        avg_mar = np.mean(mouth_aspect_ratios)

        if avg_mar >= 0.5:  # Threshold for yawning (adjust for your case)
            if not yawn_detected:
                yawn_detected = True
                yawn_count += 1
                cv2.putText(frame, f'Yawn Count: {yawn_count}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)
                print(f"Yawn Count: {yawn_count}")
        else:
            yawn_detected = False

        # Drowsiness Detection (EAR)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEyeAspectRatio = eye_aspect_ratio(leftEye)
        rightEyeAspectRatio = eye_aspect_ratio(rightEye)
        ear = (leftEyeAspectRatio + rightEyeAspectRatio) / 2

        if ear < EYE_ASPECT_RATIO_THRESHOLD:
            COUNTER += 1
            if COUNTER >= EYE_ASPECT_RATIO_CONSEC_FRAMES:
                if not drowsiness_detected:
                    drowsiness_detected = True
                    drowsiness_count += 1
                    print(f"Drowsiness Count: {drowsiness_count}")
                
        else:
            COUNTER = 0
            drowsiness_detected = False

    # Display the video feed
    cv2.imshow('Live Video', frame)

    # Exit the video when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
video_capture.release()
cv2.destroyAllWindows()
