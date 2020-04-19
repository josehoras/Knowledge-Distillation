import matplotlib.pyplot as plt

def plot_loss(loss, it, it_per_epoch, smooth_loss=[], base_name='', title=''):
    fig = plt.figure(figsize=(8, 4), dpi=100)
    plt.plot(loss)
    plt.plot(smooth_loss)
    epochs = [i * int(it_per_epoch) for i in range(int(it / it_per_epoch) + 1)]
    plt.plot(epochs, [loss[i] for i in epochs], linestyle='', marker='o')
    # if len(epochs) > 1: print(smooth_loss[epochs[-2]] -  smooth_loss[epochs[-1]] )
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    # plt.ylim([0, 3])
    if base_name != '':
        fig.savefig(base_name + '.png')
        # pickle.dump([loss, smooth_loss, it], open(base_name + '-' + str(it) + '.p', 'wb'))
        #print(it)
    else:
        plt.show()
    plt.close("all")

def plot_acc(train_acc, val_acc, it, it_per_epoch, base_name='', title=''):
    fig = plt.figure(figsize=(8, 4), dpi=100)
    if it !=0:
        inter = it//(len(train_acc) -1)
        x_axis = [i * inter for i in range(len(train_acc))]
    else:
        x_axis = [0]
    plt.plot(x_axis, train_acc, label="Train")
    plt.plot(x_axis, val_acc, label="Validation")
    plt.legend()
    # epochs = [i * int(it_per_epoch) for i in range(int(it / it_per_epoch) + 1)]
    # plt.plot(epochs, [smooth_loss[i] for i in epochs], linestyle='', marker='o')
    # if len(epochs) > 1: print(smooth_loss[epochs[-2]] -  smooth_loss[epochs[-1]] )
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Iteration')
    # plt.ylim([90, 100])
    if base_name != '':
        fig.savefig(base_name + '.png')
        # pickle.dump([loss, smooth_loss, it], open(base_name + '-' + str(it) + '.p', 'wb'))
        #print(it)
    else:
        plt.show()
    plt.close("all")