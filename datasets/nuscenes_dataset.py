import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from pyquaternion import Quaternion

# Use local import for LidarPointCloud to be safe
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.utils.data_classes import LidarPointCloud

def generate_sparse_depth(points_ego, H_feat, W_feat, horizontal_fov, vertical_fov, min_depth=2.0, max_depth=42.0):
    """
    Projects 3D points in local Ego Frame to a sparse 2D depth map.
    """
    # NuScenes Ego Frame: X_nusc: Forward, Y_nusc: Left, Z_nusc: Up
    # We map to local project coords: X_lat = -Y_nusc (right), Y_fwd = X_nusc (forward), Z_up = Z_nusc
    X_lat = -points_ego[1]
    Y_fwd = points_ego[0]
    Z_up = points_ego[2]
    
    # Filter points in front of the vehicle
    mask = Y_fwd > 0.1
    X_lat, Y_fwd, Z_up = X_lat[mask], Y_fwd[mask], Z_up[mask]
    
    # Calculate range
    depth = np.sqrt(X_lat**2 + Y_fwd**2 + Z_up**2)
    
    # Filter depth range
    depth_mask = (depth >= min_depth) & (depth <= max_depth)
    X_lat, Y_fwd, Z_up, depth = X_lat[depth_mask], Y_fwd[depth_mask], Z_up[depth_mask], depth[depth_mask]
    
    # Calculate ray angles (with 1.5m camera height offset subtracted from Z_up)
    theta = np.arctan2(-X_lat, Y_fwd)
    phi = np.arctan2(Z_up - 1.5, np.sqrt(X_lat**2 + Y_fwd**2))
    
    # Map to feature map indices
    idx_u = ((horizontal_fov / 2.0 - theta) * (W_feat / horizontal_fov)).astype(np.int32)
    idx_v = ((vertical_fov / 2.0 - phi) * (H_feat / vertical_fov)).astype(np.int32)
    
    # Validate boundary indices
    valid_grid = (idx_u >= 0) & (idx_u < W_feat) & (idx_v >= 0) & (idx_v < H_feat)
    idx_u, idx_v, depth = idx_u[valid_grid], idx_v[valid_grid], depth[valid_grid]
    
    # Allocate targets
    depth_target = np.zeros((H_feat, W_feat), dtype=np.float32)
    valid_mask = np.zeros((H_feat, W_feat), dtype=np.float32)
    
    if len(depth) > 0:
        # Sort in descending order of depth so that the smallest depth overwrites larger ones (min pooling)
        sort_idx = np.argsort(depth)[::-1]
        idx_u = idx_u[sort_idx]
        idx_v = idx_v[sort_idx]
        depth = depth[sort_idx]
        
        depth_target[idx_v, idx_u] = depth
        valid_mask[idx_v, idx_u] = 1.0
        
    return depth_target, valid_mask

def generate_occupancy_grid(points_ego, H_bev=100, W_bev=100, 
                            x_range=(-20.0, 20.0), y_range=(0.0, 40.0),
                            z_range=(0.2, 2.5)):
    """
    Generates a top-down binary occupancy grid from 3D points.
    Filters out the ground surface using a height threshold.
    """
    # NuScenes Ego Frame: X_nusc: Forward, Y_nusc: Left, Z_nusc: Up
    # We map to local project coords: X_lat = -Y_nusc (right), Y_fwd = X_nusc (forward), Z_up = Z_nusc
    X_lat = -points_ego[1]
    Y_fwd = points_ego[0]
    Z_up = points_ego[2]
    
    # Filter points by boundary ranges and height (above ground, below overhead),
    # and exclude ego-vehicle body self-reflections (reflections from the AV's own roof/hood)
    mask = (X_lat >= x_range[0]) & (X_lat < x_range[1]) & \
           (Y_fwd >= y_range[0]) & (Y_fwd < y_range[1]) & \
           (Z_up >= z_range[0]) & (Z_up < z_range[1]) & \
           ~((X_lat >= -1.0) & (X_lat <= 1.0) & (Y_fwd >= 0.0) & (Y_fwd <= 3.0))
           
    X_lat, Y_fwd = X_lat[mask], Y_fwd[mask]
    
    # Convert metric position to grid index
    idx_x = ((X_lat - x_range[0]) / (x_range[1] - x_range[0]) * W_bev).astype(np.int32)
    idx_y = ((y_range[1] - Y_fwd) / (y_range[1] - y_range[0]) * H_bev).astype(np.int32)
    
    # Clip index boundaries
    idx_x = np.clip(idx_x, 0, W_bev - 1)
    idx_y = np.clip(idx_y, 0, H_bev - 1)
    
    # Set occupancy grid values
    occupancy = np.zeros((H_bev, W_bev), dtype=np.float32)
    occupancy[idx_y, idx_x] = 1.0
    
    return occupancy

class NuscenesDataset(Dataset):
    """
    Dataloader parser for the nuScenes-mini dataset.
    Extracts multi-camera logs, LiDAR point clouds, ego motion logs,
    and constructs planning labels (expert routes).
    """
    def __init__(self, version='v1.0-mini', dataroot='data/nuscenes', 
                 split='train', T_future=8, horizontal_fov=180.0, 
                 vertical_fov=40.0, H_feat=32, W_feat=96, 
                 H_bev=100, W_bev=100, downscale_factor=2):
        super().__init__()
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.dataroot = dataroot
        self.T_future = T_future
        self.horizontal_fov = np.radians(horizontal_fov)
        self.vertical_fov = np.radians(vertical_fov)
        self.H_feat = H_feat
        self.W_feat = W_feat
        self.H_bev = H_bev
        self.W_bev = W_bev
        self.downscale_factor = downscale_factor
        
        # Load map APIs
        self.maps = {}
        for map_name in ['singapore-onenorth', 'singapore-hollandvillage', 'singapore-queenstown', 'boston-seaport']:
            try:
                self.maps[map_name] = NuScenesMap(dataroot=dataroot, map_name=map_name)
            except Exception as e:
                print(f"Warning: could not load map {map_name}: {e}")
                
        # Split scenes (mini dataset splits)
        train_scenes = ['scene-0061', 'scene-0103', 'scene-0655', 'scene-0757', 'scene-0916', 'scene-1094']
        val_scenes = ['scene-0553', 'scene-0796', 'scene-1077', 'scene-1100']
        
        target_scenes = train_scenes if split == 'train' else val_scenes
        
        self.samples_list = []
        self._prepare_samples(target_scenes)
        
    def _prepare_samples(self, target_scenes):
        for scene in self.nusc.scene:
            if scene['name'] not in target_scenes:
                continue
                
            sample_tokens = []
            curr_token = scene['first_sample_token']
            while curr_token != '':
                sample_tokens.append(curr_token)
                sample = self.nusc.get('sample', curr_token)
                curr_token = sample['next']
                
            scene_data = []
            for token in sample_tokens:
                sample = self.nusc.get('sample', token)
                lidar_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
                ego_pose = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
                
                trans = np.array(ego_pose['translation'])
                rot = Quaternion(ego_pose['rotation'])
                t = ego_pose['timestamp'] / 1e6
                yaw = rot.yaw_pitch_roll[0]
                
                scene_data.append({
                    'token': token,
                    'translation': trans,
                    'rotation': rot,
                    'yaw': yaw,
                    'timestamp': t
                })
                
            N = len(scene_data)
            if N <= self.T_future + 1:
                continue
                
            # Precompute forward velocity vx, yaw rate omega, and acceleration
            for i in range(N):
                if i < N - 1:
                    dt = scene_data[i+1]['timestamp'] - scene_data[i]['timestamp']
                    R_i = scene_data[i]['rotation'].rotation_matrix
                    t_rel = R_i.T @ (scene_data[i+1]['translation'] - scene_data[i]['translation'])
                    v_x = t_rel[0] / dt # longitudinal direction (X)
                    
                    dyaw = scene_data[i+1]['yaw'] - scene_data[i]['yaw']
                    dyaw = (dyaw + np.pi) % (2.0 * np.pi) - np.pi
                    yaw_rate = dyaw / dt
                else:
                    v_x = scene_data[i-1]['v_x']
                    yaw_rate = scene_data[i-1]['yaw_rate']
                    
                scene_data[i]['v_x'] = v_x
                scene_data[i]['yaw_rate'] = yaw_rate
                
            for i in range(N):
                if i > 0:
                    dt = scene_data[i]['timestamp'] - scene_data[i-1]['timestamp']
                    acc = (scene_data[i]['v_x'] - scene_data[i-1]['v_x']) / dt
                else:
                    acc = 0.0
                scene_data[i]['acceleration'] = acc
                
            # Extract only samples that have T_future future trajectory steps
            for i in range(N - self.T_future):
                future_traj = []
                R_curr = scene_data[i]['rotation'].rotation_matrix
                t_curr = scene_data[i]['translation']
                
                for k in range(1, self.T_future + 1):
                    t_fut_global = scene_data[i+k]['translation']
                    t_fut_local = R_curr.T @ (t_fut_global - t_curr)
                    # Convert to plan coordinates: X_lat = -Y_ego, Y_fwd = X_ego
                    future_traj.append([-t_fut_local[1], t_fut_local[0]])
                    
                future_traj = np.array(future_traj, dtype=np.float32)
                
                sample = self.nusc.get('sample', scene_data[i]['token'])
                log = self.nusc.get('log', scene['log_token'])
                map_name = log['location']
                
                self.samples_list.append({
                    'token': scene_data[i]['token'],
                    'v_x': scene_data[i]['v_x'],
                    'yaw_rate': scene_data[i]['yaw_rate'],
                    'acceleration': scene_data[i]['acceleration'],
                    'future_traj': future_traj,
                    'map_name': map_name,
                    'translation': scene_data[i]['translation'],
                    'rotation': scene_data[i]['rotation'],
                    'yaw': scene_data[i]['yaw']
                })

    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, idx):
        meta = self.samples_list[idx]
        token = meta['token']
        sample = self.nusc.get('sample', token)
        
        # 1. Load front-left, front, and front-right cameras
        camera_names = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT']
        images_list = []
        intrinsics_list = []
        extrinsics_list = []
        
        for cam_name in camera_names:
            sd_token = sample['data'][cam_name]
            sd_record = self.nusc.get('sample_data', sd_token)
            img_path = os.path.join(self.dataroot, sd_record['filename'])
            
            img = Image.open(img_path).convert('RGB')
            if self.downscale_factor > 1:
                w, h = img.size
                img = img.resize((w // self.downscale_factor, h // self.downscale_factor))
                
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            images_list.append(img_tensor)
            
            # Load intrinsics (and scale based on downscaling)
            calib_token = sd_record['calibrated_sensor_token']
            calib_record = self.nusc.get('calibrated_sensor', calib_token)
            K = np.array(calib_record['camera_intrinsic'], dtype=np.float32)
            if self.downscale_factor > 1:
                K[0, :] /= self.downscale_factor
                K[1, :] /= self.downscale_factor
            intrinsics_list.append(torch.from_numpy(K))
            
            # Load and invert extrinsics (Sensor->Ego transforms to Ego->Sensor)
            q_sensor_to_ego = Quaternion(calib_record['rotation'])
            t_sensor_to_ego = np.array(calib_record['translation'])
            
            R_ego_to_sensor = q_sensor_to_ego.rotation_matrix.T
            t_ego_to_sensor = -R_ego_to_sensor @ t_sensor_to_ego
            
            ext = np.zeros((3, 4), dtype=np.float32)
            ext[:, :3] = R_ego_to_sensor
            ext[:, 3] = t_ego_to_sensor
            
            extrinsics_list.append(torch.from_numpy(ext))
            
        images = torch.stack(images_list, dim=0)
        intrinsics = torch.stack(intrinsics_list, dim=0)
        extrinsics = torch.stack(extrinsics_list, dim=0)
        
        # 2. Load and transform LiDAR points
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_record = self.nusc.get('sample_data', lidar_token)
        lidar_path = os.path.join(self.dataroot, lidar_record['filename'])
        
        pc = LidarPointCloud.from_file(lidar_path)
        
        # Transform points to Ego Frame
        calib_token = lidar_record['calibrated_sensor_token']
        calib_record = self.nusc.get('calibrated_sensor', calib_token)
        q_lidar_to_ego = Quaternion(calib_record['rotation'])
        t_lidar_to_ego = np.array(calib_record['translation'])
        
        points_ego = q_lidar_to_ego.rotation_matrix @ pc.points[:3, :] + t_lidar_to_ego[:, np.newaxis]
        
        # 3. Generate auxiliary depth and occupancy targets
        depth_target, depth_mask = generate_sparse_depth(
            points_ego, self.H_feat, self.W_feat, self.horizontal_fov, self.vertical_fov
        )
        depth_target = torch.from_numpy(depth_target).unsqueeze(0).float()
        depth_mask = torch.from_numpy(depth_mask).unsqueeze(0).float()
        
        occupancy = generate_occupancy_grid(points_ego, self.H_bev, self.W_bev)
        occupancy_target = torch.from_numpy(occupancy).unsqueeze(0).float()
        
        # 4. Generate drivable area mask
        map_name = meta['map_name']
        nusc_map = self.maps[map_name]
        
        ego_center = np.array([20.0, 0.0, 0.0]) # Grid offset (forward view: 20m along X_ego)
        global_center = meta['rotation'].rotate(ego_center) + meta['translation']
        
        patch_box = (global_center[0], global_center[1], 40.0, 40.0)
        patch_angle = meta['yaw'] * 180.0 / np.pi
        
        drivable = nusc_map.get_map_mask(patch_box, patch_angle, layer_names=['drivable_area'], canvas_size=(self.H_bev, self.W_bev))
        drivable_target = torch.from_numpy(drivable).float()
        
        ego_state = torch.tensor([meta['v_x'], meta['yaw_rate'], meta['acceleration']], dtype=torch.float32)
        future_traj = torch.tensor(meta['future_traj'], dtype=torch.float32)
        
        return {
            'images': images,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'ego_state': ego_state,
            'future_traj': future_traj,
            'depth_target': depth_target,
            'depth_mask': depth_mask,
            'occupancy_target': occupancy_target,
            'drivable_target': drivable_target
        }
