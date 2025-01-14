import os
import random

import torch 
import numpy as np
import pandas as pd
import open3d as o3d

from personal_mlops.data import PCTreeDataset


def visualize_tree(tree: str | torch.Tensor | np.ndarray):
    if isinstance(tree, str):
        tree = pd.read_csv(tree, sep=' ', header=None)
        tree = tree.values[:,:3]
    elif isinstance(tree, torch.Tensor):
        tree = tree.cpu().numpy()
    
    assert tree.shape[1] == 3, 'Input data must have 3 columns for x, y, and z coordinates.'

    o3d_tree = o3d.geometry.PointCloud()
    o3d_tree.points = o3d.utility.Vector3dVector(tree)
    o3d.visualization.draw_geometries([o3d_tree])


if __name__ == '__main__':
    data_path = r'D:\Data\mlops_jan_2025\Dataset_pointcloud\Dataset_tree\2023-01-09_5_1_37'
    dataset = PCTreeDataset(data_path)
    
    visualize_tree(random.choice(dataset))
    