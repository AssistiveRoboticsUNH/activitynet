import tkinter as tk 
from tkinter import Image, filedialog
from datetime import timedelta
import cv2 
import os  
from PIL import Image, ImageTk 
import argparse
from tkinter import messagebox

def video_to_images(file_path): 
    if not os.path.exists(file_path):
        print(f'File not found: {file_path}')
        return []
    
    video=cv2.VideoCapture(file_path)
    imgs=[]
    while(video.isOpened()):
        ret, frame = video.read()
        if ret==False: break
        imgs.append(frame)
    video.release()
    return imgs
 

class VideoPlayer:
    def __init__(self, root, window_width=640, window_height=580):
        self.root = root
        self.root.title("CARL Video Segment Labeler 1.0")
        self.window_width = window_width
        self.window_height = window_height  
         
        self.root.geometry(f"{self.window_width+20}x{self.window_height+180}")
        # self.root.resizable(False, False)

        self.canvas = tk.Canvas(root, bg="black", width=self.window_width, height=self.window_height)
        self.canvas.grid(row=0, column=0, columnspan=5)
 
        # self.btn_open = tk.Button(root, text="Open Video", command=self.open_video)
        self.btn_open = tk.Button(
            root,
            text="Open Video",
            font=("Arial", 12, "bold"),
            bg="#212233",
            fg="white",
            command=self.open_video,
        ) 
        self.btn_open.grid(row=1, column=0, padx=0, pady=0) 

        
        self.segment_info_label = tk.Label(root, text="Segment starts: None\nSegment ends: None")
        self.segment_info_label.grid(row=1, column=1, padx=0, pady=0, sticky='w')
 
        self.btn_play = tk.Button(
            root,
            text="Play Segment",
            font=("Arial", 12, "bold"),
            bg="#2196F3",
            fg="white",
            command=self.play_segment,
        ) 
        self.btn_play.grid(row=1, column=2, padx=0, pady=0)
         


        #create a dropdown with three options, good, bad, other
        self.segment_label = tk.StringVar()
        self.segment_label.set("Good")
        self.segment_label_dropdown = tk.OptionMenu(root, self.segment_label, "Good", "Bad", "Other")
        self.segment_label_dropdown.grid(row=1, column=3, padx=10, pady=0)

        self.btn_save = tk.Button(
            root,
            text="Save Segment",
            font=("Arial", 12, "bold"),
            bg="#144336",
            #set suitable light color for save button 
            # bg="#524400",
            fg="white",
            command=self.segment_save,
        )
        self.btn_save.grid(row=1, column=4, padx=0, pady=0) 
 

        self.slider = tk.Scale(root, from_=0, to=100, orient=tk.HORIZONTAL, command=self.seek_video)
        self.slider.grid(row=2, column=0, columnspan=5, sticky='we', padx=10, pady=0)

        # self.data_box = tk.Text(root, height=7, width=45, wrap=tk.WORD, font=("Arial", 12))
        # self.data_box.grid(row=3, column=0, columnspan=4, padx=0, pady=0) 

        #add a save button
        self.btn_save2 = tk.Button(
            root,
            text="Save All",
            font=("Arial", 12, "bold"), 
            bg="#4CAF50",
            fg="white",
            command=self.segment_save_all,
            height=2,
        )
        self.btn_save2.grid(row=3, column=4, padx=0, pady=0)

        # show list of segments
        self.listbox = tk.Listbox(root, height=7, width=45, font=("Arial", 12))
        self.listbox.grid(row=3, column=0, columnspan=4, padx=0, pady=0)

        # self.listbox.insert(tk.END, "Segment starts: None")
        # self.listbox.insert(tk.END, "Segment ends: None")

        # double click listener to listbox
        self.listbox.bind("<Double-1>", self.segment_double_click)

   
        self.canvas.focus_set()  # Set focus to the canvas
        self.canvas.bind("<space>", self.toggle_playback) 
        self.canvas.bind("s", self.record_start)
        self.canvas.bind("e", self.record_end)
        self.canvas.bind("r", self.segment_reset)
        self.canvas.bind("d", self.segment_delete_last)

        # bind canvas with arrow keys
        self.canvas.bind("<Left>", self.seek_left)
        self.canvas.bind("<Right>", self.seek_right)


        self.video_source = None
        self.cap = None
        self.paused = False
        self.frame = None
        self.current_frame = 0

        self.recording_frame_start=None
        self.recording_frame_end=None
        self.playing_segment=False
        self.is_ready=False
        self.segments=[]
        self.is_replay=False

    

    def record_start(self, event):
        if not self.is_ready: return

        self.paused=True
        # if self.recording_frame_start is not None and self.recording_frame_end is not None:
        #     res=messagebox.askokcancel("Confirmation", "Start a new segment?")
        #     if not res:
        #         return
            
        # self.segment_reset(event)

        self.recording_frame_start=self.current_frame
        # self.recording_frame_end=None
        print(f"Recording frame {self.recording_frame_start} to {self.recording_frame_end}")
        

        # update segment info
        self.segment_info_label.config(text=f"Segment starts: {self.recording_frame_start}\nSegment ends: {self.recording_frame_end}")
    
    def record_end(self, event):
        if not self.is_ready: return
        self.paused=True

        if self.recording_frame_start is None:
            messagebox.showinfo("Info", "Please start recording first")
            return

        if self.current_frame <= self.recording_frame_start:
            messagebox.showinfo("Info", "End frame should be greater than start frame")
            return

        self.recording_frame_end=self.current_frame
        print(f"Recording frame {self.recording_frame_start} to {self.recording_frame_end}")
        

        self.segment_info_label.config(text=f"Segment starts: {self.recording_frame_start}\nSegment ends: {self.recording_frame_end}")

    def segment_double_click(self, event):
        print("Double clicked") 
        index = self.listbox.curselection() 
        data=self.listbox.get(index)
        print(data)
        s,e,c=data[1:-2].split(",")
        s,e=int(s),int(e)
        print(s,e,c)
        self.recording_frame_start=s
        self.recording_frame_end=e
        self.segment_label.set(c)
        self.is_replay=True
        self.play_segment()
        #regain focus
        self.canvas.focus_set()

    def segment_delete_last(self, event):
        if not self.is_ready: return

        result = messagebox.askokcancel("Confirmation", "Delete the last segment?")
        if not result:
            return
        if len(self.segments) > 0:
            self.segments.pop()
            print("Last segment deleted")
        else:
            print("No segments to delete")
        
        print(f'Segments: {self.segments}')
        self.segment_show()

    def segment_show(self):
        #clear listbox
        self.listbox.delete(0, tk.END)
        # self.data_box.insert(tk.END, f"Segment starts: {self.recording_frame_start}\nSegment ends: {self.recording_frame_end}\n\nSegments:\n")
        for seg in self.segments:
            # self.data_box.insert(tk.END, f"{seg}\n")
            self.listbox.insert(tk.END, f"{seg}")

    def segment_reset(self, event):
        if not self.is_ready: return

        self.recording_frame_start=None
        self.recording_frame_end=None
        self.segment_label.set("Good")
        self.segment_info_label.config(text=f"Segment starts: {self.recording_frame_start}\nSegment ends: {self.recording_frame_end}")
        print("Segment reset")
 

    def play_segment(self):
        if not self.is_ready: return

        self.paused=True
        if self.recording_frame_start is None or self.recording_frame_end is None:
            messagebox.showinfo("Info", "Please record a segment first")
            return

        self.current_frame = self.recording_frame_start 
        self.paused = False
        self.playing_segment=True
        self.play_video()

    def segment_save(self):
        if not self.is_ready: return

        self.paused=True
        if self.recording_frame_start is None or self.recording_frame_end is None:
            messagebox.showinfo("Info", "Please record a segment first")
            return
        
        seg=(self.recording_frame_start, self.recording_frame_end, self.segment_label.get())

        confirm = messagebox.askokcancel("Confirmation", f"Save segment: {seg}?")
        if not confirm:
            return

        self.segments.append(seg) 
        print(f'Segments: {self.segments}')
        self.segment_reset(None)
        self.segment_show()

    def segment_save_all(self):
        if not self.is_ready: return

        if len(self.segments) == 0:
            messagebox.showinfo("Info", "No segments to save")
            return

        confirm = messagebox.askokcancel("Confirmation", f"Save all segments?")
        if not confirm:
            return

        #append segments to log file.
        with open("segments_log.txt", "a") as f:
            f.write(f"file: {self.video_source}\n")
            for seg in self.segments:
                f.write(f"{seg}\n")
            f.write("\n")

        messagebox.showinfo("Info", "Segments saved to file")
        print("Segments saved to file")
        self.segments=[]
        self.segment_show()

        
    def toggle_playback(self, event):
        if not self.is_ready: return
        if self.paused:
            self.paused = False
            self.play_video()
        else:
            self.paused = True

    def seek_video(self, value):
        if not self.is_ready: return
        self.current_frame = int(value)
        if self.paused:
            self.show_current_frame()

    def seek_left(self, event):
        if not self.is_ready: return
        self.current_frame -= 1
        if self.current_frame < 0:
            self.current_frame = 0
        self.slider.set(self.current_frame)
        self.paused = True
        self.show_current_frame()
    
    def seek_right(self, event):
        if not self.is_ready: return
        self.current_frame += 1
        self.slider.set(self.current_frame)
        self.paused = True 
        self.show_current_frame()

    def open_video(self):
        file_path = filedialog.askopenfilename()
        if file_path: 
            self.play_this_video_file(file_path)
    
    def play_this_video_file(self, file_path):
        self.video_source = file_path
        self.is_ready=True
        file_name = os.path.basename(file_path)
        # update title
        self.root.title(f"CARL Video Segment Labeler 1.0 - {file_name}") 
        self.imgs = video_to_images(file_path) 
        self.current_frame = 0
        self.slider.config(from_=0, to=len(self.imgs)-1)
        self.paused = False

        #reset
        self.recording_frame_start=None
        self.recording_frame_end=None
        self.playing_segment=False  
        self.play_video()

    def show_current_frame(self):
        frame = self.imgs[self.current_frame].copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.window_width, self.window_height))
        self.frame = ImageTk.PhotoImage(Image.fromarray(frame))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.frame)

    def play_video(self):
        if not self.paused and self.current_frame < len(self.imgs): 
            self.show_current_frame()
            self.slider.set(self.current_frame) 
            self.current_frame += 1 
            self.root.after(10, self.play_video) 

            if self.playing_segment and self.current_frame > self.recording_frame_end:
                self.paused=True
                self.playing_segment=False
                print("Segment ended")

            if self.is_replay and self.current_frame > self.recording_frame_end:
                self.paused=True
                self.is_replay=False
                self.recording_frame_start=None
                self.recording_frame_end=None
                print("Segment replay ended")
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="Video file path")
    args = parser.parse_args()

    root = tk.Tk()
 
    if args.video and os.path.exists(args.video):
        print(f"Playing video: {args.video}")
        player = VideoPlayer(root)
        player.play_this_video_file(args.video)
    else:
        print("Starting empty video player...")  
        player = VideoPlayer(root)
    
    root.mainloop()
