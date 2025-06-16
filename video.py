import cv2
import mediapipe as mp
import numpy as np
import torch
import smplx
import os
import trimesh
import pyrender

# Setup MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Video paths
video_path = r"D:\SAIR_LAB\smpl\video testing\dance.mp4"
output_path = r"D:\SAIR_LAB\smpl\video testing\output_dance_smplx.mp4"

# Load SMPL-X model
model_folder = r"D:\SAIR_LAB\smpl\models_smplx_v1_1\models"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = smplx.create(
    model_path=model_folder,
    model_type='smplx',
    gender='neutral',
    num_betas=10,
    num_expression_coeffs=10,
    num_pca_comps=6,
    ext='npz'
).to(device)

# Video reader/writer setup
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Cannot open video: {video_path}")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width * 2, frame_height))

# Setup Pyrender scene & camera
scene = pyrender.Scene()
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=frame_width / frame_height)
cam_pose = np.eye(4)
cam_pose[:3, 3] = [0, 0, 3]
scene.add(camera, pose=cam_pose)

light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
scene.add(light, pose=cam_pose)

renderer = pyrender.OffscreenRenderer(viewport_width=frame_width, viewport_height=frame_height)

# Landmark to SMPL-X input conversion
def get_keypoints_2d(landmarks, frame_shape):
    keypoints = np.zeros((33, 2))
    for i, lm in enumerate(landmarks.landmark):
        keypoints[i] = [lm.x * frame_shape[1], lm.y * frame_shape[0]]
    return keypoints

# SMPL-X fitting with neutral pose
def fit_smplx_model():
    betas = torch.zeros((1, 10), dtype=torch.float32, device=device)
    expression = torch.zeros((1, 10), dtype=torch.float32, device=device)
    global_orient = torch.zeros((1, 3), dtype=torch.float32, device=device)
    body_pose = torch.zeros((1, 21, 3), dtype=torch.float32, device=device)
    face_pose = torch.zeros((1, 3), dtype=torch.float32, device=device)
    left_hand_pose = torch.zeros((1, 6), dtype=torch.float32, device=device)
    right_hand_pose = torch.zeros((1, 6), dtype=torch.float32, device=device)

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

    vertices = output.vertices[0].detach().cpu().numpy()
    joints = output.joints[0].detach().cpu().numpy()
    return vertices, joints

# Render mesh using pyrender
def render_mesh(vertices):
    mesh_color = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.9]
    tri_mesh = trimesh.Trimesh(vertices=vertices, faces=model.faces, vertex_colors=mesh_color)
    mesh = pyrender.Mesh.from_trimesh(tri_mesh)

    # Clear previous meshes
    for node in list(scene.mesh_nodes):
        scene.remove_node(node)

    scene.add(mesh)
    color, _ = renderer.render(scene)
    return cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    blank = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        keypoints_2d = get_keypoints_2d(results.pose_landmarks, frame.shape)
        vertices, joints = fit_smplx_model()
        blank = render_mesh(vertices)

        # Optional: draw projected joints (coarse)
        for joint in joints:
            x, y = int(joint[0] * 100 + frame_width // 2), int(-joint[1] * 100 + frame_height // 2)
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

    combined = np.hstack((frame, blank))
    out.write(combined)
    # cv2.imshow("SMPL-X Output", combined)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# Cleanup
cap.release()
out.release()
renderer.delete()
cv2.destroyAllWindows()
pose.close()

print(f"Finished processing. Output saved to {output_path}")
