import math
import cv2
import time
import mediapipe as mp
import matplotlib.pyplot as plt

# Initialize mediapipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=2)
mp_drawing = mp.solutions.drawing_utils

# Helper function to calculate angles
def calculateAngle(p1, p2, p3):
    x1, y1, _ = p1
    x2, y2, _ = p2
    x3, y3, _ = p3
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 360
    return angle

# Main processing function
def process_frame(frame):
    global time_hey, time_Go, flag_time, wrist_GO, wrist_hey

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(frame_rgb)

    if not result.pose_landmarks:
        return frame

    h, w, _ = frame.shape
    joints_2d = [(int(lm.x * w), int(lm.y * h)) for lm in result.pose_landmarks.landmark]
    joints_3d = [(int(lm.x * w), int(lm.y * h), lm.z * w) for lm in result.pose_landmarks.landmark]

    mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Pose classification (simplified with no real depth)
    label = 'Unknown Pose'
    color = (0, 0, 255)

    if len(joints_3d) > mp_pose.PoseLandmark.LEFT_SHOULDER.value:
        left_elbow_angle = calculateAngle(
            joints_3d[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            joints_3d[mp_pose.PoseLandmark.LEFT_ELBOW.value],
            joints_3d[mp_pose.PoseLandmark.LEFT_WRIST.value]
        )

        left_shoulder_angle = calculateAngle(
            joints_3d[mp_pose.PoseLandmark.LEFT_ELBOW.value],
            joints_3d[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            joints_3d[mp_pose.PoseLandmark.LEFT_HIP.value]
        )

        if 60 < left_elbow_angle < 130 and 75 < left_shoulder_angle < 110:
            label = "Hey"
            color = (0, 255, 0)

        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    return frame

# Read from video file
video_path = 'D:/SAIR_LAB/smpl/video testing/videoplayback.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Video file could not be opened.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame = process_frame(frame)
    cv2.imshow("Pose Estimation", processed_frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
