import numpy as np
import torch
import pyrender
import trimesh
import cv2
import smplx
import os
import joblib
from tqdm import tqdm


# set display environment variable
os.environ['PYOPENGL_PLATFORM'] = 'egl'

data = joblib.load("./output/dance_1/vibe_output_smooth.pkl")

person_key = 1
person_data = data[person_key]

frames = person_data["pose"].shape[0]

# load custom shape params
custom_betas = person_data["betas"]

# 1. initialize smplx body model
human_model_folder = r'./human_model_files'
device = torch.device("cpu")
smplx_layer = smplx.create(
    human_model_folder,
    model_type='smplx',
    gender="NEUTRAL",
    use_pca=False,
    use_face_contour=True,
    batch_size=frames
).to(device)


# 2. get smplx mesh
shape = np.tile(custom_betas[0], (frames, 1)) if len(custom_betas.shape) > 1 else custom_betas # use custom shape params
with torch.no_grad():
    smplx_output = smplx_layer(
        global_orient=torch.from_numpy(person_data["pose"][:, :3]).float(),
        body_pose=torch.from_numpy(person_data["pose"][:, 3:66]).float(),
        betas=torch.from_numpy(shape).float()
    )

smplx_root_position = smplx_output.joints[:, 0].cpu().numpy()
transl = person_data["joints3d"][:, 0, :].copy()
transl[:, 1] = -transl[:, 1]
transl[:, 2] = -transl[:, 2]
transl = (transl - smplx_root_position).reshape(frames, 1, 3)
vertices = smplx_output.vertices.cpu().numpy() + transl

renderer = pyrender.OffscreenRenderer(1080, 720)

scene = pyrender.Scene()
camera = pyrender.PerspectiveCamera(
    yfov=np.arctan(24/100)*2,  # 24mm 传感器高度
    aspectRatio=1080/720
)

camera_pose = np.eye(4)
camera_pose[2, 3] = 10
scene.add(camera, pose=camera_pose)

light = pyrender.DirectionalLight()
scene.add(light, pose=np.eye(4))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter('./output_video/vibe_output.mp4', fourcc, 30, (1080, 720))

# Rotation matrix to rotate 180 degrees around Y-axis (to face front)
R_y_180 = np.array([
    [-1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
])

for frame in tqdm(range(frames)):
    mesh = trimesh.Trimesh(
        vertices=vertices[frame],
        faces=smplx_layer.faces
    )
    # mesh.visual.vertex_colors = [128, 128, 128, 255]
    mesh = pyrender.Mesh.from_trimesh(mesh)
    mesh_node = scene.add(mesh, pose=R_y_180)
    
    # renderer = pyrender.OffscreenRenderer(1080, 720)
    color, _ = renderer.render(scene)
    video_writer.write(color)
    scene.remove_node(mesh_node)

video_writer.release()