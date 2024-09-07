from torch import optim
import torch.nn as nn
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import sys
import random
import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
from sklearn.metrics import precision_score, recall_score
# exp_name = "RMSprop_lr2E-2_epoch100"
# exp_name = "RMSprop_lr5E-2_ravel_3_layer_small_model"
exp_name = "debug"
exp_name = "filtered_big_350_2E-4"
exp_name = "filtered_float32_350_1E-4"
exp_name = "toy_38"
batch_size = 1
epochs = 800
change_epochs=100
lr = 1E-4
myseed = 0  # set a random seed for reproducibility
np.random.seed(myseed)
torch.manual_seed(myseed)
random.seed(myseed)
os.environ['PYTHONHASHSEED'] = str(myseed)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed(myseed)
    torch.cuda.manual_seed_all(myseed)
if(sys.argv[1] == '0'):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
elif(sys.argv[1] == '1'):
    torch.cuda.set_device(1)
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
elif(sys.argv[1] == '2'):
    torch.cuda.set_device(2)
    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
else:
    device = 'cpu'
print(f'DEVICE: {device}')
from model.unet_model_n_v2 import UNet
from utils.dataset import PDNDataset
net = UNet(n_channels=3, n_classes=1,bilinear=False)
net.to(device=device)
#net.load_state_dict(torch.load("save_models/{}_best_model.pth".format(exp_name), map_location=device))
# net.double()

# train_net(net, device)
# def train_net(net, device, epochs=4, batch_size=1, lr=5E-5):
# 加载训练集

pdn_dataset = PDNDataset()

train_loader = torch.utils.data.DataLoader(dataset=pdn_dataset,
                                            batch_size=batch_size, 
                                            shuffle=True, num_workers = 4)
valid_pdn_dataset = PDNDataset("test")
valid_loader = torch.utils.data.DataLoader(dataset=valid_pdn_dataset,
                                                batch_size=1, 
                                                shuffle=False, num_workers = 4)
# optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
#optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
optimizer = optim.Adam(net.parameters(), lr=lr)
print("Trainloader size: ", len(train_loader))
print("Validloader size: ", len(valid_loader))
criterion = nn.L1Loss()
best_loss = float('inf')
x = [ i for i in range(epochs)]
train_loss = []
valid_loss = []
train_f1 = []
valid_f1 = []
pbar = tqdm(range(epochs))

for epoch in pbar:
    if epoch == 200:
        lr = 1E-5
    net.train()
    t_loss = 0.0
    t_f1 = 0.0
    for image, label in train_loader:
        # print(image.shape)
        # print(label.shape)
        optimizer.zero_grad()
        image = image.to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.float32)
        pred = net(image)
        # print(pred.shape)
        loss = criterion(pred,label)
        MAE_loss = loss #record MAE loss
        t_loss += loss.item()
        pred = pred.detach().cpu().numpy().squeeze()
        label = label.detach().cpu().numpy().squeeze()
        pred = pred.ravel()
        label = label.ravel()
        # pred_top_k_idx = pred.argsort()[::-1][0:int(len(pred) / 10)]
        # pred_bot_k_idx = pred.argsort()[::-1][int(len(pred) / 10):]
        # label_top_k_idx = label.argsort()[::-1][0:int(len(label) / 10)]
        # label_bot_k_idx = label.argsort()[::-1][int(len(label) / 10):]
        # tp = len(list(set(pred_top_k_idx) & set(label_top_k_idx)))
        # fp = len(list(set(pred_top_k_idx) & set(label_bot_k_idx)))
        # fn = len(list(set(pred_bot_k_idx) & set(label_top_k_idx)))
        # Calculate the threshold for the top 10% of pixels
        pred_threshold = np.max(pred) * 0.9
        label_threshold = np.max(label) * 0.9
        print(f"pred_threshold: {pred_threshold:.5f} label_threshold: {label_threshold:.5f}")
        # Obtain indices of pixels above and below the threshold in both prediction and label arrays
        pred_top_k_idx = np.where(pred >= pred_threshold)[0]
        pred_bot_k_idx = np.where(pred < pred_threshold)[0]
        label_top_k_idx = np.where(label >= label_threshold)[0]
        label_bot_k_idx = np.where(label < label_threshold)[0]

        # Calculate true positives, false positives, and false negatives
        tp = len(np.intersect1d(pred_top_k_idx, label_top_k_idx))
        fp = len(np.intersect1d(pred_top_k_idx, label_bot_k_idx))
        fn = len(np.intersect1d(pred_bot_k_idx, label_top_k_idx))
        tn = len(np.intersect1d(pred_bot_k_idx, label_bot_k_idx))
        precision = float(tp) / (tp + fp)
        recall = float(tp) / (tp + fn)
        f1_score = 2 * precision * recall / (precision + recall + 1E-10)
        f1_loss = (1-f1_score)
        t_f1 += f1_score
        loss = MAE_loss + f1_loss

        print(f"mapsize:{len(pred.ravel()):d}  TP:{tp:d}  FP: {fp:d}  FN: {fn:d}  TN:{tn:d} precision:{precision:.5f}  recall:{recall:.5f}")
        pbar.set_description(f"[ Train | {epoch + 1:03d}/{epochs:03d} ] loss = {loss.item():.5f} MAE_loss = {MAE_loss.item():.5f} F1_loss = {f1_loss:.5f}")
        print('\n')
        # print('Loss/train', loss.item()

        # if loss < best_loss:
        #     best_loss = loss
        #     torch.save(net.state_dict(), 'best_model.pth')
        loss.backward()
        optimizer.step()
        #break ## to make them train one just one data point
    print('epoch done')
    net.eval()
    v_loss = 0.0
    v_f1 = 0.0
    for image, label in valid_loader:
        image = image.to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.float32)
        pred = net(image)
        loss = criterion(pred, label)
        # print("MAE Loss: ", loss.item())
        v_loss += loss.item()
        pred = pred.detach().cpu().numpy().squeeze()
        label = label.detach().cpu().numpy().squeeze()
        pred = pred.ravel()
        label = label.ravel()
        # pred_top_k_idx = pred.argsort()[::-1][0:int(len(pred) / 10)]
        # pred_bot_k_idx = pred.argsort()[::-1][int(len(pred) / 10):]
        # label_top_k_idx = label.argsort()[::-1][0:int(len(label) / 10)]
        # label_bot_k_idx = label.argsort()[::-1][int(len(label) / 10):]
        # # print(len(pred_top_k_idx))
        # tp = len(list(set(pred_top_k_idx) & set(label_top_k_idx)))
        # fp = len(list(set(pred_top_k_idx) & set(label_bot_k_idx)))
        # fn = len(list(set(pred_bot_k_idx) & set(label_top_k_idx)))
        # tn = len(list(set(pred_bot_k_idx) ^ set(label_bot_k_idx)))
        # print("True Positive: ", tp)
        # print("False Positive: ", fp)
        # print("False Negative: ", fn)
        # print("True Negative: ", )
        # Calculate the threshold for the top 10% of pixels
        pred_threshold = np.max(pred) * 0.9
        label_threshold = np.max(label) * 0.9
        # Obtain indices of pixels above and below the threshold in both prediction and label arrays
        pred_top_k_idx = np.where(pred >= pred_threshold)[0]
        pred_bot_k_idx = np.where(pred < pred_threshold)[0]
        label_top_k_idx = np.where(label >= label_threshold)[0]
        label_bot_k_idx = np.where(label < label_threshold)[0]

        # Calculate true positives, false positives, and false negatives
        tp = len(np.intersect1d(pred_top_k_idx, label_top_k_idx))
        fp = len(np.intersect1d(pred_top_k_idx, label_bot_k_idx))
        fn = len(np.intersect1d(pred_bot_k_idx, label_top_k_idx))
        precision = float(tp) / (tp + fp)
        # print("Precision: ", precision)
        recall = float(tp) / (tp + fn)
        # print("Recall: ", recall)
        f1_score = 2 * precision * recall / (precision + recall + 1E-10)
        v_f1 += f1_score
        # print("F1 Score: ", f1_score)
    # if(v_loss < best_loss):
    #     best_loss = v_loss
    #     torch.save(net.state_dict(), "save_models/{}_best_model.pth".format(exp_name))
    train_loss.append(t_loss/len(train_loader))
    train_f1.append(t_f1/len(train_loader))
    valid_loss.append(v_loss/len(valid_loader))
    valid_f1.append(v_f1/len(valid_loader))
    #torch.save(net.state_dict(), "save_models/{}_best_model.pth".format(exp_name))
print('train done')
torch.save(net.state_dict(), "save_models/{}_best_model.pth".format(exp_name))
print("Result:")
print(v_loss)
print(v_f1)
plt.subplot(2, 2, 1)
plt.plot(x, train_loss)
plt.xlabel("epoch")
plt.title("Train Loss")
# plt.savefig("images/{}/train_loss.png".format(exp_name))
plt.subplot(2, 2, 2)
plt.plot(x, train_f1)
plt.xlabel("epoch")
plt.title("Train F1 Score")
# plt.savefig("images/{}/train_f1.png".format(exp_name))
plt.subplot(2, 2, 3)
plt.plot(x, valid_loss)
plt.xlabel("epoch")
plt.title("Validation Loss")
# plt.savefig("images/{}/valid_loss.png".format(exp_name))
plt.subplot(2, 2, 4)
plt.plot(x, valid_f1)
plt.xlabel("epoch")
plt.title("Validation F1 Score")
# plt.savefig("images/{}/valid_f1.png".format(exp_name))
plt.tight_layout()
plt.savefig("images/{}.png".format(exp_name))

# if __name__ == "__main__":

