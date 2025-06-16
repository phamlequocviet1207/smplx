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

# Create the SMPL-X model
model = smplx.create(
    model_path=model_folder,
    model_type='smplx',
    gender='neutral',
    num_betas=10,
    num_expression_coeffs=10,
    num_pca_comps=6,
    ext='npz'
)

# Generate random shape and expression parameters
betas = torch.randn([1, model.num_betas], dtype=torch.float32)
expression = torch.randn([1, model.num_expression_coeffs], dtype=torch.float32)

# Define pose parameters
global_orient = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
body_pose = torch.zeros([1, 21, 3], dtype=torch.float32)
face_pose = torch.zeros([1, 3], dtype=torch.float32)
left_hand_pose = torch.zeros([1, 6], dtype=torch.float32)
right_hand_pose = torch.zeros([1, 6], dtype=torch.float32)

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

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Failed to open webcam.")

# Initialize pyrender scene and offscreen renderer
scene = pyrender.Scene(ambient_light=[0.5, 0.5, 0.5], bg_color=[0.2, 0.2, 0.2])
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=720/480)
camera_pose = np.eye(4)
camera_pose[:3, 3] = [0.0, 0.0, 2.5]
camera_pose[:3, :3] = np.array([
    [0.87, 0.0, 0.5],
    [0.0, 1.0, 0.0],
    [-0.5, 0.0, 0.87]
])
scene.add(camera, pose=camera_pose)

# Add multiple lights for better 3D effect
light1 = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0)
light1_pose = np.eye(4)
light1_pose[:3, 3] = [1, 1, 2]
scene.add(light1, pose=light1_pose)
light2 = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=4.0)
light2_pose = np.eye(4)
light2_pose[:3, 3] = [-1, -1, 2]
scene.add(light2, pose=light2_pose)

renderer = pyrender.OffscreenRenderer(viewport_width=720, viewport_height=480)

# Create output directory for debug frames
output_dir = "render_debug"
os.makedirs(output_dir, exist_ok=True)
frame_count = 0

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from webcam.")
            break

        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = True
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            # Draw landmarks on frame
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Get landmark coordinates
            landmarks = results.pose_landmarks.landmark
            h, w, _ = frame.shape

            # Update right elbow pose (SMPL-X index 18, MediaPipe landmark 14)
            right_elbow_idx = 14
            right_wrist_idx = 16
            right_shoulder_idx = 12
            if (right_elbow_idx < len(landmarks) and right_wrist_idx < len(landmarks) and
                right_shoulder_idx < len(landmarks) and
                landmarks[right_elbow_idx].visibility > 0.5 and
                landmarks[right_wrist_idx].visibility > 0.5 and
                landmarks[right_shoulder_idx].visibility > 0.5):
                elbow = landmarks[right_elbow_idx]
                wrist = landmarks[right_wrist_idx]
                shoulder = landmarks[right_shoulder_idx]
                # Compute 3D vectors (scale z by 1000 as it's in relative units)
                vec1 = np.array([(shoulder.x - elbow.x) * w, (shoulder.y - elbow.y) * h, (shoulder.z - elbow.z) * 1000])  # Elbow to shoulder
                vec2 = np.array([(wrist.x - elbow.x) * w, (wrist.y - elbow.y) * h, (wrist.z - elbow.z) * 1000])  # Elbow to wrist
                # Compute angle using dot product
                cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6)
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                # Adjust angle for SMPL-X (positive for flexion)
                angle = np.pi - angle  # Invert to match SMPL-X Z-axis flexion
                angle = np.clip(angle, 0.0, np.pi)  # Clip to [0, pi]
                body_pose[0, 18, 2] = angle
                # print(f"Frame {frame_count}: Right elbow angle = {angle:.2f} radians, "
                #       f"Landmarks: shoulder=({shoulder.x:.2f}, {shoulder.y:.2f}, {shoulder.z:.2f}), "
                #       f"elbow=({elbow.x:.2f}, {elbow.y:.2f}, {elbow.z:.2f}), "
                #       f"wrist=({wrist.x:.2f}, {wrist.y:.2f}, {wrist.z:.2f})")
            else:
                body_pose[0, 18, 2] = 0.0
                # print(f"Frame {frame_count}: Resetting right elbow angle to 0.0 (low visibility)")

        # Forward pass to generate vertices and joints
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

        # Center vertices at origin
        if vertices.size > 0:
            centroid = vertices.mean(axis=0)
            vertices -= centroid
            v_min, v_max = vertices.min(axis=0), vertices.max(axis=0)
            # print(f"Frame {frame_count}: Vertex range = {v_min} to {v_max}, Centroid = {centroid}")
        else:
            # print(f"Frame {frame_count}: No vertices generated!")
            continue

        # Create trimesh object
        vertex_colors = np.ones([vertices.shape[0], 4]) * [0.5, 0.5, 0.5, 1.0]
        tri_mesh = trimesh.Trimesh(vertices=vertices, faces=model.faces, vertex_colors=vertex_colors)

        # Update scene with new mesh
        for node in list(scene.get_nodes()):
            if isinstance(node, pyrender.Node) and isinstance(node.mesh, pyrender.Mesh):
                scene.remove_node(node)
        mesh = pyrender.Mesh.from_trimesh(tri_mesh)
        scene.add(mesh)

        # Render to image
        try:
            color, depth = renderer.render(scene)
            color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            # cv2.imwrite(os.path.join(output_dir, f"frame_{frame_count:04d}.png"), color_bgr)
            cv2.imshow('SMPL-X Render', color_bgr)
            # print(f"Frame {frame_count}: Rendered successfully, saved to {output_dir}/frame_{frame_count:04d}.png")
        except Exception as e:
            # print(f"Frame {frame_count}: Rendering failed: {e}")
            continue

        # Display webcam feed
        cv2.imshow('Webcam', frame)
        frame_count += 1
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    renderer.delete()