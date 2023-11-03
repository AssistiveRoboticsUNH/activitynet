import cv2                                # state of the art computer vision algorithms library
import numpy as np                        # fundamental package for scientific computing
import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures
import pyrealsense2 as rs                 # Intel RealSense cross-platform open-source API
from IPython.display import clear_output 
import datetime as dt



date=dt.datetime.now().strftime("%Y-%m-%d_%H-%M")

tosave= f'activity_{date}.avi'

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(tosave, fourcc, 20.0, (640, 480))


pipe = rs.pipeline()
profile = pipe.start()
try:
  while True:
    frames = pipe.wait_for_frames()
    color_frame = frames.get_color_frame() 

    image=np.asanyarray(color_frame.get_data())
    image= cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    out.write(image)

    cv2.namedWindow('Example', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Example', image)
    key = cv2.waitKey(1)
    # Press esc or 'q' to close the image window
    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()
        break

finally:
    pipe.stop()
    cv2.destroyAllWindows()

out.release()

print('saved to=>',tosave)
