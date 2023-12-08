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


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
eating_data=[]
max_length=30
cwd = os.getcwd()
X_train = None
y_train = None
X_test = None
y_Test = None


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


def processinput(filename, dir):
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
    return 0


def toTorch():
    eating_data2=[]
    for data in eating_data:
        if len(data)>=max_length: eating_data2.append(data[:max_length])

    X = np.array(eating_data2)
    y=np.ones(len(X))
    y[len(eating_data2):]=0

    X=torch.from_numpy(X).float()
    y=torch.from_numpy(y).reshape(-1,1)

    torchdat = separate_test_and_train(X, y)

    print(torchdat.X_train)
    print(torchdat.y_train)


def separate_test_and_train(X, y):
    ids=np.arange(X.shape[0])
    np.random.shuffle(ids)
    train_ids=ids[:int(len(ids)*0.8)]
    test_ids=ids[int(len(ids)*0.8):]
    X_train, y_train=X[train_ids], y[train_ids]
    X_test, y_test=X[test_ids], y[test_ids]
    return TorchData(X_train, y_train, X_test, y_test)


def createmodel():
    print()


def main(args):
    file = args['file']
    dir = args['directory']
    if file is None and dir is None: 
        print("No arguments specified, terminating program...")
        return
    
    valid_input = processinput(file, dir)
    if valid_input != -1: toTorch()
    else: print("Invalid input, terminating program...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Play video and save frame ids for activity recognition')
    parser.add_argument('-f','--file', help='data file', required=False) 
    parser.add_argument('-d','--directory', help='directory name', default=None)
    args = vars(parser.parse_args())
    main(args)