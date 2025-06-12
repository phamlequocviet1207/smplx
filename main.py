import os
import torch
import smplx
import pyrender
import trimesh
import numpy as np

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

# Debug: Print available joint-related information
# print("Number of body joints:", model.NUM_BODY_JOINTS)
# print("Number of face joints:", model.NUM_FACE_JOINTS)
# print("Number of hand joints:", model.NUM_HAND_JOINTS)
# print("Total number of joints:", model.NUM_JOINTS)
# print("Joint regressor shape:", model.J_regressor.shape if model.J_regressor is not None else "Not available")

# Generate random shape and expression parameters
betas = torch.randn([1, model.num_betas], dtype=torch.float32)
expression = torch.randn([1, model.num_expression_coeffs], dtype=torch.float32)

# Define pose parameters
global_orient = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)  # No rotation, upright position
body_pose = torch.zeros([1, 21, 3], dtype=torch.float32)  # 21 body joints
face_pose = torch.zeros([1, 3], dtype=torch.float32)  # 3 face joints
left_hand_pose = torch.zeros([1, 6], dtype=torch.float32)  # 6 PCA components for left hand
right_hand_pose = torch.zeros([1, 6], dtype=torch.float32)  # 6 PCA components for right hand

# Apply pose based on your joint list
# Example: Bend right elbow (index 14 in your list → SMPL-X index 19)
body_pose[0, 16, 2] = 15  # Right elbow, Z-axis rotation
# Example: Slight jaw open (index 0-10, approx. jaw)
# face_pose[0, 1] = -0.5  # Y-axis rotation for jaw

# Define mapping (adjust indices based on your 54-joint setup)
body_mapping = {
    # Mediapipe -> SMPLX
    23: 1,   # Left Hip
    24: 2,   # Right Hip
    # No    Spine1          3	        
    25: 4,   # Left Knee
    26: 5,   # Right Knee
    27: 7,   # Left Ankle
    # No    Spine2          6
    28: 8,    # Right Ankle
    # No    Spine3          9
    29: 10, # Left Foot
    30: 11, # Right Foot
    # 12	Neck	        No
    # 13	Left Collar	    No
    # 14	Right Collar	No
    # 15	Head	        No
    11: 16,  # Left Shoulder
    12: 17,  # Right Shoulder
    13: 18,  # Left Elbow
    14: 19,  # Right Elbow
    15: 20,  # Left Wrist
    16: 21   # Right Wrist
    
    
}
# hand_mapping = {0: 20, 4: 22, 8: 25, 12: 28, 16: 31, 20: 34}  # Left hand

# Forward pass to generate vertices and joints with pose
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

# Debug: Print joint positions shape
print("Joint positions shape:", output.joints.shape)

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

# Visualize the model
pyrender.Viewer(scene, use_raymond_lighting=True)