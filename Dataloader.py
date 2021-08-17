import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt
import math
import matplotlib
import matplotlib.patches as patches
import enum
from collections import namedtuple

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.prediction import PredictHelper
from pyquaternion import Quaternion
from nuscenes.eval.common.utils import quaternion_yaw

DATAROOT = '/h/cjhzhang/stuff/conditional-scene-generation/data/sets/nuscenes'
PATCH_DIM = 100
CANVAS_DIM = 1000
LAYERS = ['drivable_area', 'lane_divider']

nusc = NuScenes(version='v1.0-mini', dataroot=DATAROOT, verbose=True)
map_bs = NuScenesMap(dataroot=DATAROOT, map_name='boston-seaport')
map_sh = NuScenesMap(dataroot=DATAROOT, map_name='singapore-hollandvillage') 
map_so = NuScenesMap(dataroot=DATAROOT, map_name='singapore-onenorth') 
map_sq = NuScenesMap(dataroot=DATAROOT, map_name='singapore-queenstown') 
helper = PredictHelper(nusc)

NuscDataOutput = namedtuple('NuscDataOutput', 'ego_pose vehicles map_mask')
BatchedFrames = namedtuple('BatchedFrames', 'batched_map batched_vehicles batch_ids')

class VehicleDim(enum.IntEnum):
    X = 0
    Y = 1
    LENGTH = 2
    WIDTH = 3
    YAW = 4
    VEL = 5
    

class NuscData(torch.utils.data.Dataset):
    def __init__(self, is_train):
        self.is_train = is_train
        self.samples = self.get_samples()
        self.ego_poses = self.get_ego_poses()
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        ego_pose = self.ego_poses[idx]
        
        vehicle_anns = self.get_vehicle_anns(sample)
        vehicles = self.token_to_vector(vehicle_anns)
        map_mask = self.get_map(sample, ego_pose) 
        
        return NuscDataOutput(ego_pose, vehicles, map_mask)
            
    def get_samples(self):
        """Returns the list of samples based on the split value and dataset version"""
        
        # determine correct string to split the data on
        split = {
            'v1.0-trainval': {True: 'train', False: 'val'},
            'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
        }[nusc.version][self.is_train]

        # filters the appropriate scene names
        scenes = create_splits_scenes()[split]
        
        # collects samples with matching scene names
        samples = [samp for samp in nusc.sample if
                   nusc.get('scene', samp['scene_token'])['name'] in scenes]

        return samples
    
    def get_ego_poses(self):
        """Returns the list of ego poses that correspond to every sample"""
        ego_poses = []
        existing_samp = []
        
        for i, samp_data in enumerate(nusc.sample_data):
            samp = nusc.get('sample', samp_data['sample_token'])
            
            # add ego poses from key frames for each sample without duplicates
            if samp_data['is_key_frame'] == True and samp in self.samples and samp not in existing_samp:
                ego_poses.append(nusc.ego_pose[i])
                existing_samp.append(samp)
                
        return ego_poses      
    
    def get_vehicle_anns(self, sample):
        """Return a list of vehicle annotation tokens that appear in a given sample"""
        vehicle_anns = []
        
        for token in sample['anns']:
            actor = nusc.get('sample_annotation', token)
            
            # if 'vehicle' == actor['category_name'].split(".")[0]:
            if actor['category_name'] == 'vehicle.car':
                vehicle_anns.append(token)
                
        return vehicle_anns
    
    def get_map(self, sample, ego_pose):
        """Return the map_mask for a given sample and ego pose"""
        
        # get location for sample
        scene = nusc.get('scene', sample['scene_token'])
        log = nusc.get('log', scene['log_token'])
        location = log['location']
        
        # use the right map
        nusc_map = {
            'boston-seaport': map_bs,
            'singapore-hollandvillage': map_sh,
            'singapore-onenorth': map_so,
            'singapore-queenstown': map_sq
        }[location]
        
        # ego_pose['translation']
        x, y = ego_pose['translation'][:2]
        
        patch_box = (x, y, PATCH_DIM, PATCH_DIM)
        patch_angle = 0 
        layer_names = LAYERS
        canvas_size = (CANVAS_DIM, CANVAS_DIM)
        map_mask = nusc_map.get_map_mask(patch_box, patch_angle, layer_names, canvas_size)
        
        return map_mask
        
        
    def token_to_vector(self, vehicle_anns):
        """Returns a list of vectors that represent the vehicles in the sample.
        vector: [x_location, y_location, length, width, yaw, velocity]"""
        
        vehicles = []
        for token in vehicle_anns:
            actor = nusc.get('sample_annotation', token)
            x_loc, y_loc = actor['translation'][:2]
            length, width = actor['size'][:2]
            yaw = quaternion_yaw(Quaternion(actor['rotation']))
            velocity = helper.get_velocity_for_agent(actor['instance_token'], actor['sample_token'])
            
            vehicles.append([x_loc, y_loc, length, width, yaw, velocity])
        
        return vehicles

def plot_vehicle(ax, vehicle, ego_pose, scale):
    """Plots a vehicle and its velocity onto the map """
    width = vehicle[VehicleDim.WIDTH]*scale
    length = vehicle[VehicleDim.LENGTH]*scale
    raw_x = vehicle[VehicleDim.X] - ego_pose['translation'][0]
    raw_y = vehicle[VehicleDim.Y] - ego_pose['translation'][1]
    center_x = (raw_x + PATCH_DIM/2) * scale
    center_y = (raw_y + PATCH_DIM/2) * scale
    x = center_x - width/2
    y = center_y - length/2
    
    rect = patches.Rectangle((x, y), width, length, linewidth=1, edgecolor='r', facecolor='none')
    transform = matplotlib.transforms.Affine2D().rotate_around(center_x, center_y, vehicle[VehicleDim.YAW]) + ax.transData
    rect.set_transform(transform)    
    ax.add_patch(rect)
    
    if vehicle[VehicleDim.VEL] is not None:
        end_x = center_x + (vehicle[VehicleDim.VEL] * scale) * math.cos(vehicle[VehicleDim.YAW])
        end_y = center_y + (vehicle[VehicleDim.VEL] * scale) * math.sin(vehicle[VehicleDim.YAW])
        ax.plot([center_x, end_x], [center_y, end_y], 'c')

        dataset = NuscData(is_train=True)

def get_plot(sample):
    """ Generate the plot for a given sample from NuscData
    sample: a sample from NuscData
    """
    scale = (CANVAS_DIM/PATCH_DIM)
    ego_pose = sample[0]
    map_mask = sample[2]

    fig, ax = plt.subplots()
    ax.set_ylim([0, CANVAS_DIM])   # set the bounds to be 10, 10
    ax.set_xlim([0, CANVAS_DIM])
    ax.imshow(map_mask[0])

    for vehicle in sample[1]:
        plot_vehicle(ax, vehicle, ego_pose, scale)

    plt.show()


def collate_fn(output_list):
    # stack the map layers for each sample
    batched_map = torch.stack([torch.tensor(output.map_mask) for output in output_list])

    # concatenate the vehicles for each sample
    # have an additional tensor that indicates which batch each vehicle belong to
    batched_vehicles = torch.cat([torch.tensor(output.vehicles) for output in output_list], 0)
    
    batch_ids = []
    for i, output in enumerate(output_list):
        ids = [i for j in range(len(output.vehicles))]
        batch_ids.extend(ids)
        
    batch_ids = torch.tensor(batch_ids)
    return BatchedFrames(batched_map, batched_vehicles, batch_ids)

if __name__ == '__main__':
    dataset = NuscData(is_train=True)
    sample = dataset[44]
    get_plot(sample)
