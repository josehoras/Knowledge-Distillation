from tqdm import tqdm
from my_mnist_loader import *
import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from models import *
from plot_funcs import *

# example = 4
# img = X_train[example].reshape(28, 28)
# plt.imshow(img, cmap='Greys')
# # plt.show()
# print("Label: ", y_train[example])

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

# Set output directory and create if needed
output_dir = "teacher_linear_model/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Hyperparameters
batch_size = 100
lr = 1e-3

# Loss function
loss_fn = nn.CrossEntropyLoss()

# Create data loader
X_train, y_train, X_val, y_val, X_test, y_test = my_load_data_wrapper()

train_data = CustomDataset(X_train, y_train)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
val_data = CustomDataset(X_val, y_val)
val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True)
test_data = CustomDataset(X_test, y_test)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

# Hyperparameters
lrs = [5e-4]
dropouts = [0.4]
epochs = 10

models = {}
old_val_acc = 0
for lr in lrs:
    for dropout in dropouts:
        # title = 'lr=' + str(lr)
        title = 'dropout p=' + str(dropout)
        print("\n", title, "\n")
        # Instantiate a new network
        net = linear_net(dropout=dropout).to(device)
        # Create optimizer
        optimizer = Adam(net.parameters(), lr=lr)
        optimizer.zero_grad()
        # Start training
        val_acc = []
        train_acc = []
        train_loss = [-np.log(1.0 / 10)]  # loss at iteration 0

        # print(len(train_data), len(train_loader))
        it_per_epoch = len(train_loader)
        it = 0
        for epoch in range(epochs):
            for features, labels in tqdm(train_loader):
                scores = net(features)
                loss = loss_fn(scores, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
                if it % 100 == 0:
                    train_acc.append(evaluate(net, train_loader, max_ex=100))
                    val_acc.append(evaluate(net, val_loader))
                    plot_loss(train_loss, it, it_per_epoch, base_name=output_dir + "loss_"+title, title=title)
                    plot_acc(train_acc, val_acc, it, it_per_epoch, base_name=output_dir + "acc_"+title, title=title)
                it += 1
        #perform last book keeping
        train_acc.append(evaluate(net, train_loader, max_ex=100))
        val_acc.append(evaluate(net, val_loader))
        plot_loss(train_loss, it, it_per_epoch, base_name=output_dir + "loss_"+title, title=title)
        plot_acc(train_acc, val_acc, it, it_per_epoch, base_name=output_dir + "acc_"+title, title=title)
        print(val_acc[-1])
        models[title] = {'model': net,
                         'model_state_dict': net.state_dict(),
                         'optimizer_state_dict': optimizer.state_dict(),
                         'loss_hist': train_loss,
                         'lr':lr,
                         'p':dropout,
                         'val_acc': val_acc[-1]}

for key in models.keys():
    print("for lr: %s, val_acc: %s" % (models[key]['lr'], models[key]['val_acc']))
    # print(key)

val_accs = [models[key]['val_acc'] for key in models.keys()]
xs = [models[key]['p'] for key in models.keys()]
keys = [key for key in models.keys()]

print(val_accs)
print(lrs)
print(keys)
print("Best model is model %s" % np.argmax(val_accs))
# Plot summary
fig = plt.figure(figsize=(8, 4), dpi=100)
plt.scatter(xs, val_accs)
plt.title("{0} Epochs".format(epochs))
plt.ylabel('Validation accuracy')
# plt.xlabel('Learning rate')
plt.xlabel('dropout')
# plt.xscale('log')
# plt.xlim([9e-5, 5e-1])
fig.savefig(output_dir + 'summary_{0}epochs.png'.format(epochs))

best_key = keys[np.argmax(val_accs)]
print(best_key)
best_model = models[best_key]['model']

# model.load_state_dict(checkpoint['model_state_dict'])
# Evaluate test set
train_acc = evaluate(best_model, train_loader)
print("\nTrain accuracy: %.2f%%" % train_acc)
val_acc = evaluate(best_model,val_loader)
print("Validation accuracy: %.2f%%" % val_acc)
test_acc = evaluate(best_model, test_loader)
print("Test accuracy: %.2f%%\n" % test_acc)

torch.save({'epoch': epoch,
            'model_state_dict': best_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_hist': train_loss},
            output_dir + "modelo")

# from PIL import Image
# img = Image.open('my_number.jpg')
# arr = np.asarray(img, dtype="float32")
# arr_norm = -((np.sum(arr, axis=2)-765)/765)
# plt.imshow(arr_norm, cmap='Greys')
# plt.show()
#
# input = torch.tensor([arr_norm]).view(1, 784)
# print(input.size())
# scores = net(input)
# print("Scores: ", scores.data)
# print("Predicted class: ", torch.argmax(scores).item())