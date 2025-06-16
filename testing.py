import os
import torch
import smplx
import pyrender
import trimesh
import numpy as np
import mediapipe as mp
import cv2

# Path to the folder containing the SMPL-X model files
model_folder = r'D:\SAIR_LAB\smpl\models_smplx_v1_1\models'

# Verify the path exists
smplx_folder = os.path.join(model_folder, 'smplx')
model_file = os.path.join(smplx_folder, 'SMPLX_NEUTRAL.npz')
if not os.path.exists(model_file):
    raise FileNotFoundError(f"SMPL-X model file not found at {model_file}. Please check the path and model files.")

# Create the SMPL-X model with hand and face pose support
model = smplx.create(
    model_path=model_folder,
    model_type='smplx',
    gender='neutral',
    num_betas=10,
    num_expression_coeffs=10,
    num_pca_comps=6,  # 6 PCA components per hand
    ext='npz'
)

# Generate random shape and expression parameters
betas = torch.randn([1, model.num_betas], dtype=torch.float32)
expression = torch.randn([1, model.num_expression_coeffs], dtype=torch.float32)

# Define pose parameters
global_orient = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)  # No rotation, upright position
body_pose = torch.zeros([1, 21, 3], dtype=torch.float32)  # 21 body joints
face_pose = torch.zeros([1, 3], dtype=torch.float32)  # Jaw pose
left_hand_pose = torch.zeros([1, 6], dtype=torch.float32)  # 6 PCA components for left hand
right_hand_pose = torch.zeros([1, 6], dtype=torch.float32)  # 6 PCA components for right hand

# Define mapping (Mediapipe -> SMPL-X joint indices)
body_mapping = {
    23: 0,  # Left Hip
    24: 1,  # Right Hip
    25: 3,  # Left Knee
    26: 4,  # Right Knee
    27: 6,  # Left Ankle
    28: 7,  # Right Ankle
    29: 9,  # Left Foot
    30: 10, # Right Foot
    11: 15, # Left Shoulder
    12: 16, # Right Shoulder
    13: 17, # Left Elbow
    14: 18, # Right Elbow
    15: 19, # Left Wrist
    16: 20  # Right Wrist
}

def output_smpl():
    # Forward pass to generate vertices and joints with updated pose
    output = model(
        betas=betas,
        expression=expression,
        global_orient=global_orient,
        body_pose=body_pose,
        jaw_pose=face_pose,
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose,
        return_verts=True
    )
    vertices = output.vertices.detach().cpu().numpy().squeeze()
    joints = output.joints.detach().cpu().numpy().squeeze()

    # Create a trimesh object for visualization
    vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
    tri_mesh = trimesh.Trimesh(vertices=vertices, faces=model.faces, vertex_colors=vertex_colors)

    # Create a pyrender mesh and scene
    mesh = pyrender.Mesh.from_trimesh(tri_mesh)
    scene = pyrender.Scene()
    scene.add(mesh)

    # Add camera alignment
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = [0, 0, 3]  # Move camera back along Z-axis
    scene.add(camera, pose=camera_pose)

    # Visualize the model (non-blocking)
    pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=True)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Failed to open webcam.")


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from webcam.")
        break

    # Convert BGR to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb.flags.writeable = True  # Ensure the array is writable
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        # Draw landmarks on frame (for visualization)
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Get landmark coordinates (normalized [0,1])
        landmarks = results.pose_landmarks.landmark
        h, w, _ = frame.shape

    # Display the webcam feed
    cv2.imshow('Screen', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
pose.close()