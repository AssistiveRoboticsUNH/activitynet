import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray # Enable use of the std_msgs/Float64MultiArray message type
# from std_msgs.msg import 

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
 
import scipy.io as sio
 
from torchvision import datasets, transforms 
from torch.utils.data import DataLoader
import torchvision.models as models 
import torchvision.utils as utils  

import numpy as np
import cv2 
import pyrealsense2 as rs
import threading
from r3m import load_r3m
import omegaconf
import hydra 
import torchvision.transforms as T

device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# from torchvision.models import resnet50,ResNet50_Weights
# net=resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device)
# ### strip the last layer
# feature_extractor = torch.nn.Sequential(*list(net.children())[:-1])


r3m = load_r3m("resnet50") # resnet18, resnet34
r3m.eval()
r3m.to(device)

## DEFINE PREPROCESSING
transforms = T.Compose([T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor()]) # ToTensor() divides by 255

def extract_r3m_features(image):
  preprocessed_image = transforms(Image.fromarray(image.astype(np.uint8))).reshape(-1, 3, 224, 224)
  preprocessed_image.to(device) 
  with torch.no_grad():
    embedding = r3m(preprocessed_image * 255.0) ## R3M expects image input to be [0-255]
  return embedding.detach().cpu().numpy().squeeze()

class BcLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, out_size):
        super(BcLSTM,self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,batch_first=True)
        self.fc1 = nn.Linear(in_features=hidden_size,out_features=out_size)

    def forward(self,x):
        output,_status = self.lstm(x) 
        output = self.fc1(torch.relu(output))
        return output
    

input_size=2048
hidden_size=100
model=BcLSTM(input_size, hidden_size, 2, 6).to(device)

dir='/home/ns/matlab/inference/'
# model.load_state_dict(torch.load(dir+'resnet50sawyer_pick_bc_xyz.pt'))
# model.load_state_dict(torch.load(dir+'r3m_sawyer_pick_bc_xyz.pt'))
model.load_state_dict(torch.load(dir+'r3m_lstm_sawyer_pick_bc_xyz.pt'))



print('model loaded')


i=1
fn=f"{dir}sawyer_pick_{i}_imgs_xyz.mat"
data=sio.loadmat(fn)
actions=data['xyz']
imgs=data['imgs']


'''
Load a dataset, parse one by one image and apply inference on it.

'''


class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('bc_inference')
        self.publisher_ = self.create_publisher(Float32MultiArray, 'inference', 10)
        
        timer_period=0.2
        print('creating  timer')
        self.timer = self.create_timer(timer_period, self.capture_and_infer)
        self.closed=False
        self.i=0
        self.n=len(imgs)
        self.xs=[]

    def capture_and_infer(self):
        
        if not self.closed:
            if self.i==self.n:
                self.close()
                return 
    
            image0 = imgs[self.i]
            emb=extract_r3m_features(image0)
            fs=torch.tensor(emb).float().to(device)

            self.xs.append(torch.tensor(fs[None]).float())
            axs=torch.cat(self.xs) 
            ix=axs[None].to(device)
            
            ad=model(ix).detach().cpu().numpy().reshape(-1,6)
            paction=ad[-1]
 
            self.i = self.i +1
  
            msg = Float32MultiArray()
            msg.data = [float(pv) for pv in paction]
            # msg.data=action_data
            self.publisher_.publish(msg)

            image0= cv2.cvtColor(image0, cv2.COLOR_RGB2BGR)
            cv2.imshow("press q to close", image0)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                 self.close()


    def close(self):
        self.closed=True
        cv2.destroyAllWindows()  
        raise SystemExit



def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
