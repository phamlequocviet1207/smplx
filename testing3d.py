import cv2
import math
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt

# ------------------ MediaPipe Setup ------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# ------------------ Angle Calculation ------------------
def calculateAngle(a, b, c):
    x1, y1, _ = a
    x2, y2, _ = b
    x3, y3, _ = c
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    return angle + 360 if angle < 0 else angle

# ------------------ Pose Classification ------------------
def classifyPose(landmarks3D, image):
    ps = mp_pose.PoseLandmark
    label = 'Unknown'
    color = (0, 0, 255)

    a_le = calculateAngle(landmarks3D[ps.LEFT_SHOULDER.value],
                          landmarks3D[ps.LEFT_ELBOW.value],
                          landmarks3D[ps.LEFT_WRIST.value])
    a_ls = calculateAngle(landmarks3D[ps.LEFT_ELBOW.value],
                          landmarks3D[ps.LEFT_SHOULDER.value],
                          landmarks3D[ps.LEFT_HIP.value])
    a_re = calculateAngle(landmarks3D[ps.RIGHT_SHOULDER.value],
                          landmarks3D[ps.RIGHT_ELBOW.value],
                          landmarks3D[ps.RIGHT_WRIST.value])
    a_rs = calculateAngle(landmarks3D[ps.RIGHT_ELBOW.value],
                          landmarks3D[ps.RIGHT_SHOULDER.value],
                          landmarks3D[ps.RIGHT_HIP.value])
    a_lk = calculateAngle(landmarks3D[ps.LEFT_HIP.value],
                          landmarks3D[ps.LEFT_KNEE.value],
                          landmarks3D[ps.LEFT_ANKLE.value])
    a_rk = calculateAngle(landmarks3D[ps.RIGHT_HIP.value],
                          landmarks3D[ps.RIGHT_KNEE.value],
                          landmarks3D[ps.RIGHT_ANKLE.value])

    # Pose logic (simplified heuristics)
    if 165 < a_le < 195 and 165 < a_re < 195:
        if 80 < a_ls < 110 and 80 < a_rs < 110:
            if (165 < a_lk < 195 and 90 < a_rk < 120) or (165 < a_rk < 195 and 90 < a_lk < 120):
                label = 'Warrior II'
            elif 160 < a_lk < 195 and 160 < a_rk < 195:
                label = 'T Pose'
    if ((165 < a_lk < 195 and (315 < a_rk < 335 or 25 < a_rk < 45)) or
        (165 < a_rk < 195 and (315 < a_lk < 335 or 25 < a_lk < 45))):
        label = 'Tree Pose'

    if label != 'Unknown':
        color = (0, 255, 0)

    cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

# ------------------ Matplotlib Setup ------------------
plt.ion()
fig = plt.figure(figsize=(7.2, 4.8))  # 720x480 pixels at 100 DPI
ax = fig.add_subplot(111, projection='3d')


def plot3D_live(ax, landmarks3D, connections):
    ax.clear()
    coords = np.array(landmarks3D)

    for conn in connections:
        start, end = conn
        ax.plot([coords[start][2], coords[end][2]],  # Z
                [coords[start][0], coords[end][0]],  # X
                [coords[start][1], coords[end][1]],  # Y
                'r-')
    ax.scatter(coords[:, 2], coords[:, 0], coords[:, 1], c='blue')

    ax.set_xlabel('Z (Depth)')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    ax.set_title("Live 3D Pose")
    ax.view_init(elev=10, azim=10)
    plt.draw()
    plt.pause(0.001)

# ------------------ Video Input ------------------
video_path = 'D:/SAIR_LAB/smpl/video testing/videoplayback.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError("Cannot open video file.")

prev_time = time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (720, 480))
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Extract 3D coordinates
        landmarks3D = []
        for lm in results.pose_landmarks.landmark:
            landmarks3D.append((lm.x * w, lm.y * h, lm.z * w))  # z is in meters scaled to pixels

        classifyPose(landmarks3D, frame)
        plot3D_live(ax, landmarks3D, mp_pose.POSE_CONNECTIONS)

    # Show FPS
    curr_time = time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 60), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    # Show 2D video frame
    cv2.imshow("Pose Detection", frame)

    if cv2.waitKey(10) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
