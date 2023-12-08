import numpy as np 
import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import argparse
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import glob
from tqdm import tqdm
from collections import OrderedDict, defaultdict
from IPython.display import clear_output
import time
import torch
import torch.nn as nn
import torch.optim as optim
import random


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
eating_data=[]
max_length=30
cwd = os.getcwd()
X_train = None
y_train = None
X_test = None
y_test = None
model = None

class LSTM(nn.Module):
    
    def __init__(self,input_dim,hidden_dim,output_dim,layer_num, seq_len):
        super(LSTM,self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lstm = torch.nn.LSTM(input_dim,hidden_dim,layer_num,batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim,output_dim)
        self.bn = nn.BatchNorm1d(seq_len)
        
    def forward(self,inputs):
        x = self.bn(inputs)
        lstm_out,(hn,cn) = self.lstm(x)
        out = self.fc(lstm_out[:,-1,:])
        return out
    

class TorchData():
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test


def processfile(file):
    lines = np.loadtxt(file, dtype=str)
    ret=[]
    for line in lines:
        frame=[]
        vals = line.split(',')
        for val in vals:
            convert = float(val)
            frame.append(convert)
        ret.append(frame)
    r = np.array(ret)
    eating_data.append(r)


def processdir(dir):
    files = glob.glob(dir + "*.txt")
    for file in files:
        file = open(file, 'r')
        lines = np.loadtxt(file, dtype=str)
        ret=[]
        for line in lines:
            frame=[]
            vals = line.split(',')
            for val in vals:
                convert = float(val)
                frame.append(convert)
            ret.append(frame)
        r = np.array(ret)
        eating_data.append(r)


def processinput(filename, dir, t_model):
    global model
    if filename is not None: 
        if (os.path.isfile(filename)):
            file = open(filename, 'r')
            processfile(file)
        else: return -1
    elif dir is not None:
        if os.path.isdir(dir):
            if dir[len(dir) - 1] != '/':
                dir = dir + '/'
            processdir(dir)
        else: return -1
    if t_model is not None:
        if os.path.isfile(t_model):
            model = torch.load(t_model)
        else: return -1
    else: model = createmodel()
    return 0


def toTorch():
    global X_train, X_test, y_train, y_test
    eating_data2=[]
    for data in eating_data:
        if len(data)>=max_length: eating_data2.append(data[:max_length])

    X = np.array(eating_data2)
    y=np.ones(len(X))
    y[len(eating_data2):]=0

    print(X.shape, y.shape)

    X=torch.from_numpy(X).float()
    y=torch.from_numpy(y).reshape(-1,1)

    print(X.shape, y.shape)

    torchdat = separate_test_and_train(X, y)
    trainmodel(torchdat)
    torch.save(model.state_dict(), 'model_train_1_dec8.pth')


def separate_test_and_train(X, y):
    global X_train, X_test, y_train, y_test
    ids=np.arange(X.shape[0])
    np.random.shuffle(ids)
    train_ids=ids[:int(len(ids)*0.8)]
    test_ids=ids[int(len(ids)*0.8):]
    X_train, y_train=X[train_ids], y[train_ids]
    X_test, y_test=X[test_ids], y[test_ids]
    return TorchData(X_train, y_train, X_test, y_test)


def createmodel():
    n_hidden = 128
    n_joints = 25*2
    n_categories = 2
    n_layer = 3
    seq_len = 30
    rnn = LSTM(n_joints,n_hidden,n_categories,n_layer, seq_len)
    return rnn.to(device)


def randomTrainingExampleBatch(batch_size,flag,n_data_size_train,n_data_size_test,num=-1):
    global X_train, X_test, y_train, y_test
    if flag == 'train':
        X = X_train
        y = y_train
        data_size = n_data_size_train
    elif flag == 'test':
        X = X_test
        y = y_test
        data_size = n_data_size_test
    if num == -1:
        ran_num = random.randint(0,data_size-batch_size)
    else:
        ran_num = num
    pose_sequence_tensor = X[ran_num:(ran_num+batch_size)]
    pose_sequence_tensor = pose_sequence_tensor
    category_tensor = y[ran_num:ran_num+batch_size,:]
    return category_tensor.long(),pose_sequence_tensor


def trainmodel(torchdat):
    global model
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.0005
    optimizer = optim.SGD(model.parameters(),lr=learning_rate,momentum=0.9)

    n_iters = 20000
    print_every = 100
    plot_every = 1000
    batch_size = 4

    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []


    start = time.time()

    for iter in range(1, n_iters + 1):
        category_tensor, input_sequence = randomTrainingExampleBatch(batch_size,'train',torchdat.X_train.shape[0],torchdat.X_test.shape[0])
        input_sequence = input_sequence.to(device)
        category_tensor = category_tensor.to(device)
        category_tensor = torch.squeeze(category_tensor)

        optimizer.zero_grad()

        output = model(input_sequence)
        loss = criterion(output, category_tensor)
        loss.backward()
        optimizer.step() 

        current_loss += loss.item()

        print('iter: %d, loss: %.3f' % (iter, loss.item()))


        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

    plt.plot(all_losses)


def main(args):
    file = args['file']
    dir = args['directory']
    t_model = args['torch']
    if file is None and dir is None and t_model is None: 
        print("No arguments specified, terminating program...")
        return
    
    valid_input = processinput(file, dir, t_model)
    if valid_input != -1: toTorch()
    else: print("Invalid input, terminating program...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--file', help='data file', required=False) 
    parser.add_argument('-d','--directory', help='directory name', default=None)
    parser.add_argument('-t', '--torch', help='torch model', required=False)
    args = vars(parser.parse_args())
    main(args)