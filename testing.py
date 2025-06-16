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

# Define joint angle mappings (parent, joint, child, smplx_joint_idx, axis)
angle_mappings = [
    (12, 14, 16, 18, 2),  # Right Elbow
    (11, 13, 15, 17, 2),  # Left Elbow
    (23, 25, 27, 3, 2),   # Left Knee
    (24, 26, 28, 4, 2),   # Right Knee
    (0, 23, 25, 0, 0),    # Left Hip
    (0, 24, 26, 1, 0),    # Right Hip
    (0, 11, 13, 15, 0),   # Left Shoulder
    (0, 12, 14, 16, 0),   # Right Shoulder
]

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
camera = pyrender.PerspectiveCamera(yfov=np.pi / 2.5, aspectRatio=720/480)  # Wider FOV
camera_pose = np.eye(4)
camera_pose[:3, 3] = [0.0, 0.5, 3.5]  # Further and slightly raised
camera_pose[:3, :3] = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
])  # Straight-on view
scene.add(camera, pose=camera_pose)

# Add lights
light1 = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0)
light1_pose = np.eye(4)
light1_pose[:3, 3] = [1, 1, 2]
scene.add(light1, pose=light1_pose)
light2 = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=4.0)
light2_pose = np.eye(4)
light2_pose[:3, 3] = [-1, -1, 2]
scene.add(light2, pose=light2_pose)

renderer = pyrender.OffscreenRenderer(viewport_width=720, viewport_height=480)

# Create output directory (unused)
output_dir = "render_debug"
os.makedirs(output_dir, exist_ok=True)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = True
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Get landmark coordinates
            landmarks = results.pose_landmarks.landmark
            h, w, _ = frame.shape

            # Compute angles for multiple joints
            for parent_idx, joint_idx, child_idx, smplx_idx, axis in angle_mappings:
                if (parent_idx < len(landmarks) and joint_idx < len(landmarks) and
                    child_idx < len(landmarks) and
                    landmarks[parent_idx].visibility > 0.5 and
                    landmarks[joint_idx].visibility > 0.5 and
                    landmarks[child_idx].visibility > 0.5):
                    parent = landmarks[parent_idx]
                    joint = landmarks[joint_idx]
                    child = landmarks[child_idx]
                    vec1 = np.array([(parent.x - joint.x) * w, (parent.y - joint.y) * h, (parent.z - joint.z) * 1000])
                    vec2 = np.array([(child.x - joint.x) * w, (child.y - joint.y) * h, (child.z - joint.z) * 1000])
                    cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6)
                    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                    angle = np.pi - angle
                    angle = np.clip(angle, 0.0, np.pi)
                    body_pose[0, smplx_idx, axis] = angle
                else:
                    body_pose[0, smplx_idx, axis] = 0.0

            # Estimate global orientation
            if (23 < len(landmarks) and 24 < len(landmarks) and
                landmarks[23].visibility > 0.5 and landmarks[24].visibility > 0.5):
                left_hip = landmarks[23]
                right_hip = landmarks[24]
                hip_vec = np.array([(right_hip.x - left_hip.x) * w, (right_hip.y - left_hip.y) * h, (right_hip.z - left_hip.z) * 1000])
                yaw = np.arctan2(hip_vec[0], hip_vec[2])
                global_orient[0, 1] = yaw

        # Forward pass
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

        # Center and scale vertices
        if vertices.size > 0:
            centroid = vertices.mean(axis=0)
            vertices -= centroid
            # Normalize scale to fit within unit cube
            max_extent = np.max(np.abs(vertices), axis=0).max()
            if max_extent > 0:
                vertices /= max_extent * 1.5  # Scale down to fit view
            vertices[:, 1] += 0.5  # Raise slightly to center vertically
        else:
            continue

        # Create trimesh
        vertex_colors = np.ones([vertices.shape[0], 4]) * [0.5, 0.5, 0.5, 1.0]
        tri_mesh = trimesh.Trimesh(vertices=vertices, faces=model.faces, vertex_colors=vertex_colors)

        # Update scene
        for node in list(scene.get_nodes()):
            if isinstance(node, pyrender.Node) and isinstance(node.mesh, pyrender.Mesh):
                scene.remove_node(node)
        mesh = pyrender.Mesh.from_trimesh(tri_mesh)
        scene.add(mesh)

        # Render
        try:
            color, depth = renderer.render(scene)
            color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            cv2.imshow('SMPL-X Render', color_bgr)
        except Exception as e:
            continue

        # Display webcam
        cv2.imshow('Webcam', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    renderer.delete()