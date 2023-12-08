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

device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

from torchvision.models import resnet50,ResNet50_Weights
net=resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device)

### strip the last layer
feature_extractor = torch.nn.Sequential(*list(net.children())[:-1])

class BC(nn.Module):
    def __init__(self, input_size=2048, output_size=6):
        super(BC, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x)) 
        x=self.fc2(x)
        return x
    
model =  BC().to(device)
dir='/home/ns/matlab/inference/'
model.load_state_dict(torch.load(dir+'resnet50sawyer_pick_bc_xyz.pt'))
print('model loaded')


'''
Apply inference on live camera.
'''


class MyRealSense:
    def __init__(self):
        self.pipe = rs.pipeline()
        self.profile = self.pipe.start()

    def get_current_frame(self, scale=0.5):
        frames = self.pipe.wait_for_frames()
        color_frame = frames.get_color_frame() 

        image=np.asanyarray(color_frame.get_data())
        image= cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        image = cv2.resize(image, (int(image.shape[1]*scale), int(image.shape[0]*scale) ) , interpolation = cv2.INTER_AREA)
        return image 

    def close(self):
        self.pipe.stop()



class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('bc_inference')
        self.publisher_ = self.create_publisher(Float32MultiArray, 'inference', 10)
        self.cam=MyRealSense()
        timer_period=0.005
        print('creating  timer')
        self.timer = self.create_timer(timer_period, self.capture_and_infer)
        self.closed=False
        # msg = String()
        # msg.data = 'Hello pub: %d' % self.i
        # self.publisher_.publish(msg)

    def capture_and_infer(self):
        
        if not self.closed:
            image = self.cam.get_current_frame()
            x=torch.tensor(image.transpose(2,0,1)).float().to(device)[None]
            fs=feature_extractor(x).squeeze()
            paction=model(fs).detach().cpu().numpy()
            action_data=str(paction)
            # msg=String() 
            # print('paction=', paction)

            msg = Float32MultiArray()
            msg.data = [float(pv) for pv in paction]
            # msg.data=action_data
            self.publisher_.publish(msg)

            cv2.imshow("press q to close", image)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                self.close()

    def close(self):
        self.closed=True
        cv2.destroyAllWindows()
        self.cam.close()



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
