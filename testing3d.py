# -*- coding: utf-8 -*-
"""
Created on  Jan 2022

@author: Sia_Mahmoudi
"""

import math
from typing import List, Optional, Tuple
import dataclasses
import cv2
import time
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mediapipe.framework.formats import landmark_pb2

# presets
time_pre = time.time()
time_hey = 0
time_Go = 0
flag_time = 0
wrist_GO = 0
wrist_hey = 0

# mediapipe settings for plotting
_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5
_RGB_CHANNELS = 3

WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
BLUE_COLOR = (255, 0, 0)

# mediapipe drawing class
@dataclasses.dataclass
class DrawingSpec:
    color: Tuple[int, int, int] = WHITE_COLOR
    thickness: int = 2
    circle_radius: int = 2

# Initializing mediapipe pose class.
mp_pose = mp.solutions.pose

# Setting up the Pose function.
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

# Initializing mediapipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils 

def _normalize_color(color):
    return tuple(v / 255. for v in color)

def plot_landmarkss(landmarks_D, 
                    landmark_list: landmark_pb2.NormalizedLandmarkList,
                    connections: Optional[List[Tuple[int, int]]] = None,
                    landmark_drawing_spec: DrawingSpec = DrawingSpec(
                        color=RED_COLOR, thickness=5),
                    connection_drawing_spec: DrawingSpec = DrawingSpec(
                        color=BLACK_COLOR, thickness=5),
                    elevation: int = 10,
                    azimuth: int = 10):
    if not landmark_list:
        return
    plt.clf()  # Clear previous plot to reuse figure
    fig = plt.gcf()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=elevation, azim=azimuth)
    plotted_landmarks = {}
    for idx, landmark in enumerate(landmark_list.landmark):
        if ((landmark.HasField('visibility') and
             landmark.visibility < _VISIBILITY_THRESHOLD) or
            (landmark.HasField('presence') and
             landmark.presence < _PRESENCE_THRESHOLD)):
            continue
        if idx < len(landmarks_D): 
            ax.scatter3D(
                xs=[landmarks_D[idx]],
                ys=[landmark.x],
                zs=[-landmark.y],
                color=_normalize_color(landmark_drawing_spec.color[::-1]),
                linewidth=landmark_drawing_spec.thickness)
            plotted_landmarks[idx] = (landmarks_D[idx], landmark.x, -landmark.y)
    if connections:
        num_landmarks = len(landmark_list.landmark)
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                raise ValueError(f'Landmark index is out of range. Invalid connection '
                                 f'from landmark #{start_idx} to landmark #{end_idx}.')
            if start_idx in plotted_landmarks and end_idx in plotted_landmarks:
                landmark_pair = [
                    plotted_landmarks[start_idx], plotted_landmarks[end_idx]]
                ax.plot3D(
                    xs=[landmark_pair[0][0], landmark_pair[1][0]],
                    ys=[landmark_pair[0][1], landmark_pair[1][1]],
                    zs=[landmark_pair[0][2], landmark_pair[1][2]],
                    color=_normalize_color(connection_drawing_spec.color[::-1]),
                    linewidth=connection_drawing_spec.thickness)
    plt.draw()
    plt.pause(0.001)

def detectPose(image, pose, display=True):
    output_image = image.copy()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(imageRGB)
    height, width, _ = image.shape
    landmarks_3D = []
    landmarks_2D = []
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)
        for landmark in results.pose_landmarks.landmark:
            landmarks_3D.append((int(landmark.x * width), int(landmark.y * height),
                                 (landmark.z * width)))
            landmarks_2D.append((int(landmark.x * width), int(landmark.y * height)))
        landmarks_D = []
        if len(landmarks_2D) >= 1:
            for i in range(len(landmarks_2D)):
                if 0 < landmarks_2D[i][0] < width and 0 < landmarks_2D[i][1] < height:
                    landmarks_D.append(landmark.z * 1000)  # Scale z to approximate mm
    if display:
        plt.figure(figsize=[22,22])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
        plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
        plot_landmarkss(landmarks_D, results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
    else:
        return output_image, landmarks_3D, landmarks_2D

def calculateAngle(landmark1, landmark2, landmark3):
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 360
    return angle

def calculate_landmarks_D(joints, z_values):
    landmarks_D = []
    if len(joints) >= 1:
        for i in range(len(joints)):
            if 0 < joints[i][0] < 1280 and 0 < joints[i][1] < 720:
                landmarks_D.append(z_values[i] * 1000)  # Scale z to approximate mm
    return landmarks_D

def classifyPose(landmarks, output_image, joints, z_values, display=False):
    global time_hey, time_Go, flag_time, wrist_GO, wrist_hey
    label = 'Unknown Pose'
    color = (0, 0, 255)
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
    if (0 < joints[mp_pose.PoseLandmark.LEFT_ELBOW.value][0] < 1280 and 
        0 < joints[mp_pose.PoseLandmark.LEFT_ELBOW.value][1] < 720 and
        0 < joints[mp_pose.PoseLandmark.LEFT_WRIST.value][0] < 1280 and 
        0 < joints[mp_pose.PoseLandmark.LEFT_WRIST.value][1] < 720):
        LEFT_ELBOW_depth = z_values[mp_pose.PoseLandmark.LEFT_ELBOW.value] * 1000
        LEFT_WRIST_depth = z_values[mp_pose.PoseLandmark.LEFT_WRIST.value] * 1000
        if left_elbow_angle > 60 and left_elbow_angle < 130 and left_shoulder_angle > 75 and left_shoulder_angle < 110:
            if 50 > (abs(int(LEFT_ELBOW_depth) - int(LEFT_WRIST_depth))) > 0:
                label = 'Hey'
                time_hey = round(((time.time()) - time_pre), 2)
                wrist_hey = LEFT_WRIST_depth
            elif 400 > (abs(int(LEFT_ELBOW_depth) - int(LEFT_WRIST_depth))) > 180:
                label = "GO"
                wrist_GO = LEFT_WRIST_depth
                time_Go = round(((time.time()) - time_pre), 2)
            else:
                label = '****READY****'
            if label != 'Unknown Pose':
                color = (0, 255, 0)
            cv2.putText(output_image, label, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
            if flag_time != time_hey:
                if time_hey != 0 and time_Go != 0 and (time_Go > time_hey) and (time_Go - time_hey < 4):
                    print("*")
                    print("*")
                    print('**Order is detected*********')
                    if (time_Go - time_hey) < 1.5:
                        Velocity = round(((wrist_hey - wrist_GO) / ((time_Go - time_hey) * 1000)), 2)
                        print("Go Fast")
                        print("Velocity: ", Velocity)
                        cv2.putText(output_image, "GO FAST", (1050, 45), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                        cv2.putText(output_image, f"velocity:{Velocity} m/s", (1000, 105), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                    if (time_Go - time_hey) > 1.5:
                        Velocity = round(((wrist_hey - wrist_GO) / ((time_Go - time_hey) * 1000)), 2)
                        print("Go Slow")
                        print("Velocity: ", Velocity)
                        cv2.putText(output_image, "GO SLOW", (1050, 45), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                        cv2.putText(output_image, f"velocity:{Velocity} m/s", (1000, 105), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                    flag_time = time_hey
    if display:
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
    else:
        return output_image, time_hey, time_Go, label

# Initialize Matplotlib figure
plt.ion()
plt.figure(figsize=(10, 10))

# Open video file
video_path = r'D:\SAIR_LAB\smpl\video testing\videoplayback.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Failed to open video file at {video_path}")
    exit()

try:
    while cap.isOpened():
        ret, bgr_frame = cap.read()
        if not ret:
            print("End of video or failed to read frame")
            break

        # Convert the image from BGR into RGB format
        imageRGB = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        results = pose.process(imageRGB)

        # Perform Pose landmark detection
        bgr_frame, landmarks, joints = detectPose(bgr_frame, pose, display=False)

        if landmarks:
            # Extract z values for pose classification
            z_values = [landmark[2] / width for landmark in landmarks]  # Normalize z by width
            frame, time_hey, time_Go, _ = classifyPose(landmarks, bgr_frame, joints, z_values, display=False)
        else:
            continue

        # Calculate landmarks_D for 3D plotting
        z_values_scaled = [z * 1000 for z in z_values]  # Scale for plotting
        landmarks_D = calculate_landmarks_D(joints, z_values)
        
        # Display depth values on frame
        if len(joints) >= 1:
            for i in range(len(joints)):
                if 0 < joints[i][0] < 1280 and 0 < joints[i][1] < 720:
                    cv2.putText(bgr_frame, str(int(z_values_scaled[i])), joints[i],
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("bgr frame", bgr_frame)

        key = cv2.waitKey(10)
        if key == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    plt.close()