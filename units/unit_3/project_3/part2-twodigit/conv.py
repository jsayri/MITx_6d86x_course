import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_utils import batchify_data, run_epoch, train_model, Flatten
import utils_multiMNIST as U
path_to_data_dir = '../Datasets/'
use_mini_dataset = True

batch_size = 64
nb_classes = 10
nb_epoch = 30
num_classes = 10
img_rows, img_cols = 42, 28 # input image dimensions

# output size equation for convolution & pooling layers, to keep dimension use n_out = fun(n_in, 3, 1, 1, 1)
n_out_eq = lambda n_in, k, p, s, d: (n_in + 2 * p - d * (k - 1) - 1) / s + 1


class CNN(nn.Module):

    def __init__(self, input_dimension):
        super(CNN, self).__init__()
        # set convolution layers parameters
        n_c1 = 32 # conv kernels number
        n_c2 = 64 # conv kernels number
        n_p1 = 2 # pool number
        n_p2 = 3 # pool number
        n_fc1 = input_dimension * n_c2 / (2 * n_p1) # pool2 keep's dimension
        self.conv1 = nn.Conv2d(1, n_c1, (3, 3), padding=1) # convolutional layer, keep dimension
        self.pool1 = nn.MaxPool2d((n_p1, n_p1)) # pool layer, reduce dimension in n_p1 factor
        self.conv2 = nn.Conv2d(n_c1, n_c2, (3, 3), padding=1) # convolution layer, keep dimension
        self.pool2 = nn.MaxPool2d(n_p2, 1, 1, 1) # poll layer, keep dimensions
        self.drop = nn.Dropout() # dropout layer
        self.flatten = Flatten() # flatten function
        self.fc = nn.Linear(int(n_fc1), 64) # fully connected layer
        self.out = nn.Linear(64, 20) # output layer

    def forward(self, x):
        xc = F.relu(self.conv1(x)) # activation function ReLU after convolution
        xp = self.pool1(xc)
        xc2 = F.relu(self.conv2(xp))
        xp2 = self.pool2(xc2)
        xd = self.drop(xp2)
        xf = self.flatten(xd) # xf = F.relu(self.flatten(xd)) # not a big change
        xl = self.fc(xf) # xl = F.relu(self.fc(xf)) # not a big change
        xo = self.out(xl)
        # use model layers to predict the two digits
        out_first_digit = xo[:, 0:10]
        out_second_digit = xo[:, 10:]

        return out_first_digit, out_second_digit

def main():
    X_train, y_train, X_test, y_test = U.get_data(path_to_data_dir, use_mini_dataset)

    # Split into train and dev
    dev_split_index = int(9 * len(X_train) / 10)
    X_dev = X_train[dev_split_index:]
    y_dev = [y_train[0][dev_split_index:], y_train[1][dev_split_index:]]
    X_train = X_train[:dev_split_index]
    y_train = [y_train[0][:dev_split_index], y_train[1][:dev_split_index]]

    permutation = np.array([i for i in range(len(X_train))])
    np.random.shuffle(permutation)
    X_train = [X_train[i] for i in permutation]
    y_train = [[y_train[0][i] for i in permutation], [y_train[1][i] for i in permutation]]

    # Split dataset into batches
    train_batches = batchify_data(X_train, y_train, batch_size)
    dev_batches = batchify_data(X_dev, y_dev, batch_size)
    test_batches = batchify_data(X_test, y_test, batch_size)

    # Load model
    input_dimension = img_rows * img_cols
    model = CNN(input_dimension)
    model.cuda()

    # Train
    train_model(train_batches, dev_batches, model)

    ## Evaluate the model on test data
    loss, acc = run_epoch(test_batches, model.eval(), None)
    print('Test loss1: {:.6f}  accuracy1: {:.6f}  loss2: {:.6f}   accuracy2: {:.6f}'.format(loss[0], acc[0], loss[1], acc[1]))

if __name__ == '__main__':
    # Specify seed for deterministic behavior, then shuffle. Do not change seed for official submissions to edx
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)  # for reproducibility
    main()