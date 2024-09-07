import glob
import numpy as np
import torch
import os
from model.unet_model_n_v2 import UNet
import torch.nn as nn
import torchvision.transforms as transforms
import sys
import pandas as pd
import time
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split

def getData(current_map, eff_dist, pdn_density):
    max_value = {'current_map': 3.74722e-07, 'eff_dist_map': 61.3528, 'pdn_density': 3.0}
    min_value = {'current_map': 3.05644e-08, 'eff_dist_map': 0.0, 'pdn_density': 0.0}
    category = ["current_map", "eff_dist_map", "pdn_density"]
    data = [[]]

    for j, c in enumerate(category):
        if c == "current_map":
            df = pd.read_csv(current_map, header = None, dtype = "float32")
        elif c == "eff_dist_map":
            df = pd.read_csv(eff_dist, header = None, dtype = "float32")
        else:
            df = pd.read_csv(pdn_density, header = None, dtype = "float32")
        df = df.to_numpy()
        dim = len(df)
        df = df.ravel()
        df = df - min_value[c]
        df = df / (max_value[c] - min_value[c])
        df = df.reshape((dim, -1))
        # df = np.transpose(df)
        data[0].append(df)
    dim = len(data[0][0])
    img = np.zeros((3, dim, dim))
    for j in range(3):
        img[j] = data[0][j]
    ir_img = np.zeros((1, dim, dim)) #just all zeros
    data[0] = [img,ir_img]
    return data

class PDNDataset(Dataset):

    def __init__(self,current_map, eff_dist, pdn_density):
        super(PDNDataset).__init__()
        self.data = getData(current_map, eff_dist, pdn_density)
        
        
    def __len__(self):
        return len(self.data)
  
    def __getitem__(self,idx):
        
        img = torch.from_numpy(self.data[idx][0])
        ir_img = torch.from_numpy(self.data[idx][1])
        return img, ir_img
    
if __name__ == "__main__":

    myseed = 0  # set a random seed for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    
    # take input files
    current_map = sys.argv[1]
    pdn_density = sys.argv[2]
    eff_dist = sys.argv[3]
    xyz_sp = sys.argv[4]
    chk_pnt = sys.argv[5]
    path_to_output_predicted_ir_drop = sys.argv[6]

    device = 'cpu'
    print(f'DEVICE: {device}')

    # initialize a Model and load the model from chk_pnt
    net = UNet(n_channels=3, n_classes=1)
    net.to(device=device)
    net.load_state_dict(torch.load(chk_pnt, map_location=device))
    net.eval()

    start = time.time()

    # Predict
    pdn_dataset = PDNDataset(current_map,eff_dist,pdn_density)
    test_loader = torch.utils.data.DataLoader(dataset=pdn_dataset,
                                               batch_size=1, 
                                               shuffle=False, num_workers = 4)
    for i, (image, label) in enumerate(test_loader):
        image = image.to(device=device, dtype=torch.float32)
        pred = net(image)
        
        pred = pred.detach().cpu().numpy().squeeze()
        df = pd.DataFrame(pred)
            
        df.to_csv(path_to_output_predicted_ir_drop, encoding='utf-8', header = False, index = False)

    end = time.time()
print(end - start)
print("Prediction done.")