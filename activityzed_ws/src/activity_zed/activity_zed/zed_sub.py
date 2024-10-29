import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from zed_interfaces.msg import ObjectsStamped
import cv2
from cv_bridge import CvBridge
import os
import rclpy
from rclpy.node import Node
import imageio 
import datetime 
import getpass
import argparse
import subprocess
import threading
import time 
import yaml 

def get_is_rosbag_running_via_cli():
    result = subprocess.run(["ros2", "node", "list"], capture_output=True, text=True)
    if result.returncode == 0:
        active_nodes = result.stdout 
        # print("Active nodes:")
        # for name, namespace in active_nodes:
        #     print(f"{namespace}/{name}")
            
        if "rosbag2_player" in active_nodes:
                return True
    return False


def list_topics(metadata_path): 
    metadata= yaml.load(open(metadata_path, 'r'), Loader=yaml.FullLoader)

    topics={}
    topic_metadatas=metadata['rosbag2_bagfile_information']['topics_with_message_count']
    for topic_metadata in topic_metadatas:
        topic_name=topic_metadata['topic_metadata']['name']
        message_count=topic_metadata['message_count']
        topics[topic_name]=message_count
        
    return topics



def play_ros2_bag(bag_file_path, loop=False):
    # Prepare the ros2 bag play command
    command = ["ros2", "bag", "play", bag_file_path]
    if loop:
        command.append("--loop")

    try:
        # Start playing the bag file
        print(f"Playing ROS 2 bag file: {bag_file_path}")
        process = subprocess.Popen(command)
        
        # Wait for the process to complete (or terminate it programmatically if desired)
        process.wait()

    except KeyboardInterrupt:
        print("Interrupted, stopping playback.")
        process.terminate()
        process.wait()  # Ensure the process ends

class ZedSub(Node):

    def __init__(self, view=False, rosbag_path=None):
        super().__init__('zed_sub')
        
        topic_name_image='/zed_data_recording/zed_node_data_recording/left/image_rect_color'
        topic_name_skeleton='/zed_data_recording/zed_node_data_recording/body_trk/skeletons'
        
        number_of_frames = -1
        if rosbag_path is not None:
            metadata_path= os.path.join(rosbag_path, 'metadata.yaml')
            topics=list_topics(metadata_path)
            topic_name_image= [name for name in topics.keys() if 'image' in name][0]
            topic_name_skeleton= [name for name in topics.keys() if 'skeleton' in name][0]
            number_of_frames=topics[topic_name_image]
            
            print('Playing rosbag in a new thread')
            threading.Thread(target=play_ros2_bag, args=(rosbag_path,)).start()
            print('Waiting for rosbag to start')
            time.sleep(5)
            
            
            
            
        print('Number of frames: ',number_of_frames)
        print('Image topic: ',topic_name_image)
        print('Skeleton topic: ',topic_name_skeleton)
        
            
        
        self.isview=view
        
        print('Subscribing to: ',topic_name_image)
        print('Subscribing to: ',topic_name_skeleton)


        self.subscription = self.create_subscription(
            Image,
            topic_name_image,
            self.image_callback,
            10)
        
        self.sub_skeleton = self.create_subscription(
            ObjectsStamped,
            topic_name_skeleton,
            self.skeleton_callback,
            10)


        self.bridge = CvBridge()

        self.kp_indices=[1,2,3,4,5,6,7,8,9, 10,11,12,13,14,15,16,17, 30,31,32,33,34,35,36,37]
        self.current_points_2d=[]
        self.current_points_3d=[]

        now = datetime.datetime.now()
        time_str=now.strftime("%m_%d_%Y_%H_%M")
        
        
        # Get the current username
        username = getpass.getuser()

        self.savedir=f"/home/{username}/activitynet/videos/{time_str}/"
        
        if rosbag_path is not None:
            self.savedir=rosbag_path
            fn='_'.join( rosbag_path.split("/")[-4:] )
            self.savedir = os.path.join(self.savedir, fn)
            
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)
        
        video_path=os.path.join(self.savedir, 'zed.mp4')
        print('saving to: ',video_path)
        self.video_writer = imageio.get_writer(video_path, fps=5)
        self.si=0
        self.p2ss=[]
        self.p3ss=[]


    def skeleton_callback(self, pos_msg):
        objs=pos_msg.objects
        # print('\npose: ',len(objs), pos_msg)
        objs = pos_msg.objects
        points_2d=[]
        points_3d=[]
        for obj in objs:
            isk=obj.skeleton_available
            if isk:
                body_format=obj.body_format          #2->38 keypoints
                kps=obj.skeleton_2d.keypoints
                kps3d=obj.skeleton_3d.keypoints

                for id in self.kp_indices:
                    points_2d.append(kps[id].kp)
                    points_3d.append(kps3d[id].kp)
                # print(f'\nSkeleton: {isk} {body_format} {len(kps)} {len(kps3d)}')
                break #only one person

        self.current_points_2d=points_2d
        self.current_points_3d=points_3d
        

    def image_callback(self, image_msg):
        if image_msg is None:
            return
 
        self.si+=1
        if self.si<5:
            print('(check_5): image frame received: ',self.si)
        
        self.cv2_image = self.bridge.imgmsg_to_cv2(image_msg,
                                                    desired_encoding='passthrough')  # Preserve original encoding
        # bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        width,height=self.cv2_image.shape[1],self.cv2_image.shape[0]
        p2s=self.current_points_2d
        p3s=self.current_points_3d

        p2s_str=''
        p3s_str=''
        for i in range(len(p2s)):
            p2=p2s[i]
            p3=p3s[i]
            p2s_str+=f'{int(p2[0])},{int(p2[1])},'
            p3s_str+=f'{p3[0]},{p3[1]},{p3[2]},' 
 
            # x,y=int(width*p2[0]),int(height*p2[1])
            x,y=int(p2[0]),int(p2[1])
            if x<0 or y<0:
                continue

            original_size = (720, 1080)  # Original image size (width, height)
            new_size = (height, width)

            scale_x= new_size[0] /original_size[0]
            scale_y= new_size[1] /original_size[1]

            x=int(x*scale_x)
            y=int(y*scale_y)

            # print(width, height, x,y , p3[0], p3[1], p3[2])
            # cv2.circle(self.cv2_image, (x, y), 5, (255, 0, 0), -1)  

        # for i in range(0,len(p2s),2):
        #     x1,y1=int(p2s[i][0]),int(p2s[i][1])
        #     x2,y2=int(p2s[i+1][0]),int(p2s[i+1][1])
        #     cv2.line(self.cv2_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        self.p2ss.append(p2s_str)
        self.p3ss.append(p3s_str) 

        #draw serial number in the image
        cv2.putText(self.cv2_image, str(self.si), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        bgr_img = cv2.cvtColor(self.cv2_image, cv2.COLOR_RGB2BGR)
        self.video_writer.append_data(bgr_img)
        
        if self.isview:
            cv2.imshow('topic_image',self.cv2_image)
            if cv2.waitKey(1) == 27: 
                self.close() 
                print('UI closed')
 

    def close(self):
        print('Closing')
        with open(os.path.join(self.savedir, 'p2s.txt'), 'w') as f:
            for item in self.p2ss:
                f.write("%s\n" % item)
        with open(os.path.join(self.savedir, 'p3s.txt'), 'w') as f:
            for item in self.p3ss:
                f.write("%s\n" % item)

        with open(os.path.join(self.savedir, 'segments.txt'), 'w') as f:
            f.write('start,end,label\n') 
            
        self.video_writer.close()
        print('Video and 2D,3D keypoints saved')

        cv2.destroyAllWindows()


def check_rosbag2(node):
    while True:
        is_running=get_is_rosbag_running_via_cli()
        # print('Is rosbag2 running: ',is_running)
        time.sleep(1)
        if not is_running:
            node.close()
            print('----------node closed-----------')
            time.sleep(2)
            try:
                node.destroy_node() 
            except:
                pass 
            rclpy.shutdown()
            print('----------rclpy shutdown-----------')
            break

def main(args=None):
    rclpy.init(args=None)

    node = ZedSub(args.view, args.rosbag_path)
    threading.Thread(target=check_rosbag2, args=(node,)).start()
    

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('KeyboardInterrupt')

    finally:
        # Destroy the node explicitly
        # (optional - otherwise it will be done automatically
        # when the garbage collector destroys the node object)
        node.close()
        node.destroy_node()
        # rclpy.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Zed Sub')
    parser.add_argument('--view', action='store_true', help='View the images')
    parser.add_argument('--rosbag_path', type=str, help='Path to the rosbag directory')
    args = parser.parse_args()
    main(args)


# python3 zed_sub.py --rosbag_path /media/ns/Seagate/oct18/lauren/cam1/rosbag2_lauren/rosbag2_lauren


# python3 zed_sub.py --rosbag_path  /media/ns/Seagate/oct18/ashley/cam1/rosbag2_ashley/rosbag2_ashley
# python3 zed_sub.py --rosbag_path  /media/ns/Seagate/oct18/ashley/cam1/rosbag2_ashley_1/rosbag2_ashley_1

