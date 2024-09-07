import pandas as pd
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
# Transform

img_size = 512

test_tfm = transforms.Compose([
    # transforms.Resize((img_size, img_size)),
    # transforms.RandomHorizontalFlip(p = 0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# However, it is also possible to use augmentation in the testing phase.
# You may use train_tfm to produce a variety of images and then test using ensemble methods
train_tfm = transforms.Compose([
    # transforms.Resize((img_size, img_size)),
    # transforms.RandomHorizontalFlip(p = 0.5),
    # transforms.RandomRotation(30),
    # transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


"""## **Datasets**
Use pandas to read csv and prepare tensors
"""

def getRealData():
    max_value = {'current_map': 3.74722e-07, 'eff_dist_map': 61.3528, 'pdn_density': 3.0}
    min_value = {'current_map': 3.05644e-08, 'eff_dist_map': 0.0, 'pdn_density': 0.0}
    index = [1, 2, 3, 4, 5, 6, 17, 18]
    index = [1, 2, 3, 4, 5, 6, 11, 12, 17, 18]
    data = [[] for i in range(len(index))]
    # category = ["current_map", "eff_dist_map", "pdn_density", "ir_drop_map"]
    category = ["current_map", "eff_dist_map", "pdn_density"]

    for i in range(len(index)):
        for j, c in enumerate(category):
            df = pd.read_csv("real-circuit-data_20230615/testcase{}/{}.csv".format(index[i], c), header = None, dtype = "float32")
            df = df.to_numpy()
            dim = len(df)
            df = df.ravel()
            df = df - min_value[c]
            df = df / (max_value[c] - min_value[c])
            
            df = df.reshape((dim, -1))
            # df = np.transpose(df)
            data[i].append(df)
        df = pd.read_csv("real-circuit-data_20230615/testcase{}/ir_drop_map.csv".format(index[i]), header = None, dtype = "float32")
        df = df.to_numpy()
        data[i].append(df)
        print("real:",i)
        
        dim = len(data[i][0])
        img = np.zeros((3, dim, dim))
        ir_img = np.zeros((1, dim, dim))
        for j in range(3):
            img[j] = data[i][j]
        ir_img[0] = data[i][3]
        data[i] = [img, ir_img]
    return data

def getFakeData():
    max_value = {'current':0,'eff_dist':0,'regions':0}
    min_value = {'current':float('inf'),'eff_dist':float('inf'),'regions':float('inf')}
    max_value = {'current': 3.74722e-07, 'eff_dist': 61.3528, 'regions': 3.0}
    min_value = {'current': 3.05644e-08, 'eff_dist': 0.0, 'regions': 0.0}
    # category = ["current", "eff_dist", "pdn_density", "ir_drop_map"]
    category = ["current", "eff_dist", "regions"]
    # f_index = [12, 18, 2162, 96]
    # f_index = [12, 18, 62, 96]
    data = []
    data = [[] for i in range(100)]
    for i in range(100):
        # if(i in f_index):
        #     continue
        # data.append([])
        for j, c in enumerate(category):
            df = pd.read_csv("fake-circuit-data_20230615/current_map{:02d}_{}.csv".format(i, c), header = None, dtype = "float32")
            df = df.to_numpy()
            dim = len(df)
            df = df.ravel()
            df = df - min_value[c]
            df = df / (max_value[c] - min_value[c])
                
            df = df.reshape((dim, -1))
            # df = np.transpose(df)
            data[i].append(df)
        df = pd.read_csv("fake-circuit-data_20230615/current_map{:02d}_voltage.csv".format(i), header = None, dtype = "float32")
        df = df.to_numpy()
        data[i].append(df)
        print("fake:",i)

        dim = len(data[i][0])
        img = np.zeros((3, dim, dim))
        ir_img = np.zeros((1, dim, dim))
        for j in range(3):
            img[j] = data[i][j]
        ir_img[0] = data[i][3]
        data[i] = [img, ir_img]
    #print(max_value)
    #print(min_value)
    return data

def getMoreFakeData():

    max_value = {'current': 3.74722e-07, 'eff_dist': 61.3528, 'pdn_density': 3.0}
    min_value = {'current': 3.05644e-08, 'eff_dist': 0.0, 'pdn_density': 0.0}
    # category = ["current", "eff_dist", "pdn_density", "ir_drop_map"]
    category = ["current", "eff_dist", "pdn_density"]
    # f_index = [12, 18, 2162, 96]
    # f_index = [12, 18, 62, 96]
    data = []
    data = [[] for i in range(900)]
    for i in range(900):
        # if(i in f_index):
        #     continue
        # data.append([])
        for j, c in enumerate(category):
            df = pd.read_csv("fake-circuit-data_20230825/BeGAN_{:04d}_{}.csv".format(i+100, c), header = None, dtype = "float32")
            df = df.to_numpy()
            dim = len(df)
            df = df.ravel()
            df = df - min_value[c]
            df = df / (max_value[c] - min_value[c])
                
            df = df.reshape((dim, -1))
            # df = np.transpose(df)
            data[i].append(df)
        df = pd.read_csv("fake-circuit-data_20230825/BeGAN_{:04d}_ir_drop.csv".format(i+100), header = None, dtype = "float32")
        df = df.to_numpy()
        data[i].append(df)
        print("data1000:",i)

        dim = len(data[i][0])
        img = np.zeros((3, dim, dim))
        ir_img = np.zeros((1, dim, dim))
        for j in range(3):
            img[j] = data[i][j]
        ir_img[0] = data[i][3]
        data[i] = [img, ir_img]
    #print(max_value)
    #print(min_value)
    return data

class PDNDataset(Dataset):

    def __init__(self, mode = "train", tfm = None):
        super(PDNDataset).__init__()
        if mode == "train":
            self.data = getFakeData()
        elif mode == "train_more":
            self.data = getMoreFakeData()
        else:
            self.data = getRealData() 
        self.transform = tfm
        self.mode = mode
        
    def __len__(self):
        return len(self.data)
  
    def __getitem__(self,idx):
        # dim = len(self.data[idx][0])
        # img = np.zeros((3, dim, dim))
        # ir_img = np.zeros((1, dim, dim))
        # for i in range(3):
        #     img[i] = self.data[idx][i]
        # img = torch.from_numpy(img)
        img = torch.from_numpy(self.data[idx][0])
        # img = img.reshape(3, dim, dim)
        if(self.transform):
            # print("img")
            #img = Image.fromarray(img.astype('double'))
            img = self.transform(img)

        # ir_img[0] = self.data[idx][3]
        # ir_img = torch.from_numpy(ir_img)
        ir_img = torch.from_numpy(self.data[idx][1])
        if(self.transform):
            # print("ir_img")
            #ir_img = Image.fromarray(ir_img.astype('double'))
            ir_img = self.transform(ir_img)
        # if self.mode == "train":
        #     ir_img = Image.fromarray(ir_img.astype('float'))
        #     ir_img = self.transform(ir_img)
        # print("return")
        # print(img.shape)
        # print(ir_img.shape)
        return img, ir_img
