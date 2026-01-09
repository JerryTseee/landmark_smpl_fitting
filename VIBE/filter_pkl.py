import numpy as np
import joblib
import argparse
from scipy.ndimage import gaussian_filter1d

def smooth_pkl(input_path, output_path, sigma=2.0):
    """
    Simple Gaussian smoothing for pose data
    
    Args:
        input_path: Path to input .pkl file
        output_path: Path to save smoothed .pkl file
        sigma: Standard deviation for Gaussian kernel. Higher = smoother
    """
    
    # Load data
    print(f"Loading data from {input_path}")
    data = joblib.load(input_path)
    
    # Process each person in the data
    for person_id, person_data in data.items():
        print(f"Processing person {person_id}")
        
        # Get pose data (frames × 72 pose parameters)
        poses = person_data['pose'].copy()  # Shape: (n_frames, 72)
        

        # Apply Gaussian filter along time axis (axis=0)
        poses[:, 3:] = gaussian_filter1d(poses[:, 3:], sigma=sigma, axis=0)
        
        # Update the pose data
        data[person_id]['pose'] = poses
        
        # Optional: Also smooth joints3d for consistency
        if 'joints3d' in person_data:
            joints3d = person_data['joints3d']  # Shape: (n_frames, n_joints, 3)
            # Flatten for smoothing, then reshape
            joints3d_flat = joints3d.reshape(joints3d.shape[0], -1)
            smoothed_joints3d_flat = gaussian_filter1d(joints3d_flat, sigma=sigma, axis=0)
            data[person_id]['joints3d'] = smoothed_joints3d_flat.reshape(joints3d.shape)
        
        # Optional: Also smooth vertices if available
        if 'verts' in person_data:
            verts = person_data['verts']  # Shape: (n_frames, n_vertices, 3)
            verts_flat = verts.reshape(verts.shape[0], -1)
            smoothed_verts_flat = gaussian_filter1d(verts_flat, sigma=sigma, axis=0)
            data[person_id]['verts'] = smoothed_verts_flat.reshape(verts.shape)
    
    # Save smoothed data
    print(f"Saving smoothed data to {output_path}")
    joblib.dump(data, output_path)
    
    print(f"Done! Applied Gaussian smoothing with sigma={sigma}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smooth pose data in .pkl file")
    parser.add_argument("--input", type=str, required=True, help="Input .pkl file path")
    parser.add_argument("--output", type=str, required=True, help="Output .pkl file path")
    parser.add_argument("--sigma", type=float, default=2.0, help="Smoothing strength (higher = smoother)")
    
    args = parser.parse_args()
    
    smooth_pkl(args.input, args.output, args.sigma)