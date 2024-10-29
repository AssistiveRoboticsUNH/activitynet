import rclpy
from rclpy.node import Node
import subprocess
import os
import yaml


def get_topics_via_cli():
    result = subprocess.run(["ros2", "topic", "list"], capture_output=True, text=True)
    if result.returncode == 0:
        active_topics = result.stdout.split("\n") 
        print("Active topics:")
        for name in active_topics:
            print(f"{name}")
            # image_rect_color
        if "skeletons" in active_topics:
                return True
    return False
        
def is_ros2_bag_running(node): 
    active_nodes = node.get_node_names_and_namespaces()
    # print("Active nodes:")
    # for name, namespace in active_nodes:
    #     print(f"{namespace}/{name}")
        
    # Check for ros2 bag nodes
    for name, _ in active_nodes:
        if "rosbag2_record" in name or "rosbag2_play" in name:
            return True
    return False



def list_topics_from_metadata(bag_directory):
    print(f"Listing topics from metadata.yaml in {bag_directory}")
    # Path to the metadata.yaml file
    metadata_path = os.path.join(bag_directory, 'metadata.yaml')

    # Check if the metadata file exists
    if not os.path.isfile(metadata_path):
        print(f"No metadata.yaml found in {bag_directory}")
        return

    # Load the metadata.yaml file
    with open(metadata_path, 'r') as file:
        metadata = yaml.safe_load(file)

    # Extract topic names
    if 'topics' in metadata:
        print("Topics in the bag file:")
        for topic in metadata['topics']:
            print(topic['name'])
    else:
        print("No topics found in metadata.")

 




def list_nodes():
    rclpy.init()
    node = Node("list_nodes_example")
     
    iscli=get_topics_via_cli()
    print(f"Is CLI running: {iscli}")
    
    
    bag_directory = '/media/ns/Seagate/oct25/madison/cam1/rosbag2_madison_s_25_1/rosbag2_madison_s_25_1' 
    list_topics_from_metadata(bag_directory)

    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    list_nodes()
