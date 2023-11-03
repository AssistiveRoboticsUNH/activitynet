import numpy as np
import cv2
import matplotlib.pyplot as plt
#clear output
from IPython.display import clear_output
import time
import threading
import time


dir="F:\\OneDrive - USNH\\Documents\\ActivePresenter\\Untitled1\\Video\\"
fn="dain1.mp4"
file_path=dir+fn

video=cv2.VideoCapture(file_path)
imgs=[]
while(video.isOpened()):
    ret, frame = video.read()
    if ret==False: break
    imgs.append(frame)
video.release()




frame_no=0
closed=False
key_pressed=None 
paused=False

frame_ids=[]
def play_images(imgs):
    global frame_no
    global closed
    global key_pressed
    global paused
    global frame_ids

    info="space: pause/play \nq: quit \na: previous frame \nd: next frame\ns: save frame"

    while True:
        img=imgs[frame_no].copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = 'Frame: ' + str(frame_no)
        cv2.putText(img, text, (10, 50), font, 1, (0, 255, 255), 2, cv2.LINE_AA)
 
        y0, dy = 50, 30
        for i, line in enumerate(info.split('\n')):
            y = y0 + i*dy
            cv2.putText(img, line, (200, y ), font, 1, (0, 255, 255), 1, cv2.LINE_AA)
            # cv2.putText(img, line, (100, y ), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)


        cv2.imshow('Video', img)
        #get key pressed
        key_pressed=cv2.waitKey(1)

        
        if not paused and key_pressed & 0xFF == ord(' '):
            paused=True 
        elif paused and key_pressed & 0xFF == ord(' '):
            paused=False
        elif key_pressed & 0xFF == ord('a'):
            frame_no-=1
            if frame_no<0: frame_no=0
        elif key_pressed & 0xFF == ord('d'):
            frame_no+=1
            if frame_no>=len(imgs): frame_no=len(imgs)-1
        elif key_pressed & 0xFF == ord('s'):
            frame_ids.append(frame_no)
            print(f'frame_ids={frame_ids}')

        if key_pressed & 0xFF == ord('q'):
            closed=True
            cv2.destroyAllWindows()  
            break 

thread = threading.Thread(target=play_images, args=(imgs, ))
thread.start()

frame_no=0
while True: 
    if closed: break

    if frame_no==1:
        paused=True
    
    time.sleep(delay)

    if paused:
        # print('paused')
        pass 
    else:
        frame_no+=1

