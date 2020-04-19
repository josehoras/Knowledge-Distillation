from tqdm import tqdm
from my_mnist_loader import *
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from models import *


class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        print('data shape: ', self.features.shape)
        print('labels shape: ', self.labels.shape)

    def __getitem__(self, index):
        f = torch.tensor(self.features[index])
        l = torch.tensor(self.labels[index])
        return (f.to(device), l.to(device))

    def __len__(self):
        return len(self.labels)


def evaluate(model, dataset, max_ex=0):
    acc = 0
    N = len(dataset) * batch_size
    for i, (features, labels) in enumerate(dataset):
        scores = model(features)
        pred = torch.argmax(scores, dim=1)
        acc += torch.sum(torch.eq(pred, labels)).item()
        if max_ex != 0 and i >= max_ex:
            break
    # print(i)
    return (acc * 100 / ((i+1) * batch_size) )


### MAIN
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# Set directory to load model from
load_path = "small_linear_model/"

# Instantiate a new network
model = small_linear_net().to(device)

# Create optimizer
optimizer = Adam(model.parameters(), lr=0.0005)
optimizer.zero_grad()

checkpoint = torch.load(load_path + "modelo")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss_hist = checkpoint['loss_hist']

print(checkpoint['optimizer_state_dict'].keys())
print(checkpoint['optimizer_state_dict']['param_groups'])


model.eval()
# - or -
# model.train()

# Create data loader
X_train, y_train, X_val, y_val, X_test, y_test = my_load_data_wrapper()
batch_size = 100
train_data = CustomDataset(X_train, y_train)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
val_data = CustomDataset(X_val, y_val)
val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)
test_data = CustomDataset(X_test, y_test)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

train_acc = evaluate(model, train_loader)
print("\nTrain accuracy: %.2f%%" % train_acc)
val_acc = evaluate(model,val_loader)
print("Validation accuracy: %.2f%%" % val_acc)
test_acc = evaluate(model, test_loader)
print("Test accuracy: %.2f%%" % test_acc)