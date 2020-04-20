from tqdm import tqdm
import os
from my_mnist_loader import *
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from models import *
from plot_funcs import *


class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        # print('data shape: ', self.features.shape)
        # print('labels shape: ', self.labels.shape)

    def __getitem__(self, index):
        f = torch.tensor(self.features[index])
        l = torch.tensor(self.labels[index])
        return (f.to(device), l.to(device))

    def __len__(self):
        return len(self.labels)


def evaluate(model, dataset, max_ex=0):
    model.eval()
    acc = 0
    N = len(dataset) * batch_size
    for i, (features, labels) in enumerate(dataset):
        scores = model(features)
        pred = torch.argmax(scores, dim=1)
        acc += torch.sum(torch.eq(pred, labels)).item()
        if max_ex != 0 and i >= max_ex:
            break
    model.train()
    return (acc * 100 / ((i+1) * batch_size) )


### MAIN
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# Set directory to load model from
load_path = "teacher_linear_model/"

# Instantiate a new network
big_model = linear_net().to(device)


# Create optimizer
# optimizer = Adam(model.parameters(), lr=0.0005)
# optimizer.zero_grad()

checkpoint = torch.load(load_path + "modelo")
big_model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss_hist = checkpoint['loss_hist']

# print(checkpoint['optimizer_state_dict'].keys())
# print(checkpoint['optimizer_state_dict']['param_groups'])


big_model.eval()
# # - or -
# big_model.train()

# Create data loader
X_train, y_train, X_val, y_val, X_test, y_test = my_load_data_wrapper()
batch_size = 100
train_data = CustomDataset(X_train, y_train)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
val_data = CustomDataset(X_val, y_val)
val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True)
test_data = CustomDataset(X_test, y_test)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)


### Training the small model
softmax_op = nn.Softmax(dim=1)
# Loss functions
# loss_fn = nn.CrossEntropyLoss()
mseloss_fn = nn.MSELoss()


def my_loss(scores, targets, T = 5):
    soft_pred = softmax_op(scores / T)
    soft_targets = softmax_op(targets / T)
    loss = mseloss_fn(soft_pred, soft_targets)
    return loss


# Set output directory and create if needed
output_dir = "small_linear_model_distill_rnd/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

load=True

temperatures = [4]
lrs = [1e-5]
epochs = 10
sweep = "T"

models = {}
for temp in temperatures:
    for lr in lrs:
        if sweep=="T": title = 'T=' + str(temp)
        if sweep=="lr": title = 'lr=' + str(lr)
        print("\n", title, "\n")
        # create new student network
        small_model = small_linear_net().to(device)
        # Create optimizer
        optimizer = Adam(small_model.parameters(), lr=lr)
        # Start training
        val_acc = []
        train_acc = []
        train_loss = [0]#[-np.log(1.0 / 10)]  # loss at iteration 0
        it_per_epoch = len(train_loader)
        it = 0
        # Maybe load previous model
        if load == True:
            checkpoint = torch.load(output_dir + "modelo_data")
            small_model.load_state_dict(checkpoint['model_state_dict'])
            checkpoint['optimizer_state_dict']['param_groups'][0]['lr'] = lr
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print()
            print(checkpoint['optimizer_state_dict']['param_groups'])
            print(checkpoint.keys())
            val_acc = checkpoint['val_acc']
            train_acc = checkpoint['train_acc']
            train_loss = checkpoint['loss_hist']
            it = checkpoint['iterations']

        for epoch in range(epochs):
            for features, labels in tqdm(train_loader):
                rnd_features = torch.rand(100, 784).to(device)
                scores = small_model(rnd_features)
                targets = big_model(rnd_features)

                loss = my_loss(scores, targets, T= temp)
                # if it == 0:
                #     print(scores.size())
                #     print(scores[0])
                #     print(targets[0])
                #     soft_pred = softmax_op(scores / temp)
                #     soft_targets = softmax_op(targets / temp)
                #     print(soft_pred[0])
                #     print(soft_targets[0])
                #     loss_rep = mseloss_fn(soft_pred, soft_targets)
                #     print(loss)
                #     print(loss_rep)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
                if it % 100 == 0:
                    train_acc.append(evaluate(small_model, train_loader, max_ex=100))
                    val_acc.append(evaluate(small_model, val_loader))
                    plot_loss(train_loss, it, it_per_epoch, base_name=output_dir + "loss_"+title, title=title)
                    plot_acc(train_acc, val_acc, it, it_per_epoch, base_name=output_dir + "acc_"+title, title=title)
                    print('Iteration: %i, %.2f%%' % (it, val_acc[-1]))
                it += 1
        #perform last book keeping
        train_acc.append(evaluate(small_model, train_loader, max_ex=100))
        val_acc.append(evaluate(small_model, val_loader))
        plot_loss(train_loss, it, it_per_epoch, base_name=output_dir + "loss_"+title, title=title)
        plot_acc(train_acc, val_acc, it, it_per_epoch, base_name=output_dir + "acc_"+title, title=title)

        print(val_acc[-1])
        models[title] = {'model': small_model,
                         'model_state_dict': small_model.state_dict(),
                         'optimizer_state_dict': optimizer.state_dict(),
                         'lr': lr,
                         'T': temp,
                         'loss_hist': train_loss,
                         'train_acc': train_acc,
                         'val_acc': val_acc,
                         'iterations': it}



val_accs = [models[key]['val_acc'][-1] for key in models.keys()]
if sweep=="T":  xs = [models[key]['T'] for key in models.keys()]
if sweep=="lr": xs = [models[key]['lr'] for key in models.keys()]
keys = [key for key in models.keys()]

print(val_accs)

print(keys)
print("Best model is model %s" % np.argmax(val_accs))
# Plot summary
fig = plt.figure(figsize=(8, 4), dpi=100)
plt.scatter(xs, val_accs)
plt.title("{0} Epochs".format(epochs))
plt.ylabel('Validation accuracy')
if sweep=="T": plt.xlabel('T')
if sweep=="lr":
    plt.xlabel('Learning rate')
    plt.xscale('log')
    plt.xlim([xs[0]/2, xs[-1]*2])
fig.savefig(output_dir + 'summary_{0}epochs.png'.format(epochs))

best_key = keys[np.argmax(val_accs)]
print(best_key)
best_model = models[best_key]['model']
best_model.eval()

torch.save(models[best_key],
           output_dir + "modelo")

# {'model': small_model,
#                          'model_state_dict': small_model.state_dict(),
#                          'optimizer_state_dict': optimizer.state_dict(),
#                          'lr': lr,
#                          'T': temp,
#                          'loss_hist': train_loss,
#                          'train_acc': train_acc,
#                          'val_acc': val_acc,
#                          'iterations': it}


print("\nBig model")
train_acc = evaluate(big_model, train_loader)
print("Train accuracy: %.2f%%" % train_acc)
val_acc = evaluate(big_model,val_loader)
print("Validation accuracy: %.2f%%" % val_acc)
test_acc = evaluate(big_model, test_loader)
print("Test accuracy: %.2f%%" % test_acc)

print("\nSmall model")
train_acc = evaluate(best_model, train_loader)
print("Train accuracy: %.2f%%" % train_acc)
val_acc = evaluate(best_model,val_loader)
print("Validation accuracy: %.2f%%" % val_acc)
test_acc = evaluate(best_model, test_loader)
print("Test accuracy: %.2f%%\n" % test_acc)

from PIL import Image
img = Image.open('my_number.jpg')
arr = np.asarray(img, dtype="float32")
arr_norm = -((np.sum(arr, axis=2)-765)/765)
# plt.imshow(arr_norm, cmap='Greys')
# plt.show()

input = torch.tensor([arr_norm]).view(1, 784).to(device)
print(input.size())
scores = best_model(input)
print("Scores: ", scores.data.cpu().numpy())
print("Predicted class: ", torch.argmax(scores).item())
