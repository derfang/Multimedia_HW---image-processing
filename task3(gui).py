import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2  # Assuming you have OpenCV installed for video player (if needed)
import sounddevice as sd
from scipy.io.wavfile import write
import threading
import time
import os
import socket
import sys
import torch
from torchvision import models, transforms
import json
import torch.nn.functional as F


class VideoPlayerApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        
        self.model = models.resnet50(pretrained=True)
        self.model.eval()
        
        
        
        
        # Define fixed dimensions for video and changeable sidebar
        sidebar_width = 200  # Adjust this value for desired sidebar width
        video_width = 600
        video_height = 400

        # Set window size
        self.window.geometry(f"{video_width + sidebar_width}x{video_height}")

        # Create frames
        self.video_frame = tk.Frame(window, width=video_width, height=video_height)
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.sidebar_frame = tk.Frame(window, width=sidebar_width, height=video_height, bg="lightgray")
        self.sidebar_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create a button to open a file dialog to select a video (positioned at the bottom)
        self.btn_capture = ttk.Button(self.sidebar_frame, text="capture frame", command=self.capture_frame)

        self.lbl_filename = ttk.Label(self.sidebar_frame, text="",background="lightgray")
        self.lbl_filename.grid(column=0, row=1, sticky=tk.W,pady=5,padx=5)



        self.btn_capture.grid(column=0, row=4,pady=15,padx=5)

        # this button will select image from system
        self.btn_select_image = ttk.Button(self.sidebar_frame, text="Select Image", command=self.select_image)
        self.btn_select_image.grid(column=0, row=5,pady=5,padx=5)
        self.image_path = None
        
        
        self.btn_predict =ttk.Button(self.sidebar_frame, text="predict", command=self.predict)# use lambda to pass image address to send file func.
        self.btn_predict.grid(column=0, row=7,pady=0,padx=1)
        self.send_status = tk.StringVar()
        
        self.send_label = tk.Label(self.sidebar_frame, text="",background="lightgray",textvariable= self.send_status)
        self.send_label.grid(column=0, row=9,pady=0,padx=5)
        self.btn_predict["state"] = tk.DISABLED
        # self.checkbox1.grid(column=0, row=5,pady=5,padx=5)
        # self.checkbox2.grid(column=0, row=4,pady=5,padx=5)
        # self.checkbox3.grid(column=0, row=5,pady=5,padx=5)
        # # Ensure sidebar stays at the bottom
        # self.sidebar_frame.pack(side=tk.BOTTOM, fill=tk.Y)


        # Create a canvas for video playback
        self.canvas = tk.Canvas(self.video_frame, width=video_width, height=video_height)
        self.canvas.pack()

        # Set up the video capture device
        self.cap = None
        self.open_camera()
        # Create a label to display the filename


        # Start the Tkinter event loop
        # self.window.bind("<Escape>", self.go_back_to_first_page)
        self.window.protocol('WM_DELETE_WINDOW', self.closeapp) 
            # client ip

        
        self.window.mainloop()
    def select_image(self):
        self.image_path = filedialog.askopenfilename()
        self.lbl_filename["text"] = os.path.basename(self.image_path)
        self.btn_predict["state"] = tk.ACTIVE

    def closeapp(self):
        
        self.cap.release()
        self.window.destroy()
        
    def update_booleans(self):
        boolean1 = self.var1.get()
        # Implement your logic using boolean values here
        print(f"Checkbox 1: {boolean1}")

        # Example: Modify video playback based on checkboxes
        if boolean1:
            # Do something when Checkbox 1 is checked (e.g., adjust video brightness)
            self.btn_connect['state' ]= tk.DISABLED
            pass


    def go_back_to_first_page(self, a):
        # Destroy the video player window
        self.window.destroy()


    
    
    def start_record(self):
        self.record_thread.daemon=True
        self.record_thread.start()
        self.record_remain = 30
        self.recording_time_thread.daemon = True
        self.recording_time_thread.start()
        
    def predict(self):
        # Perform prediction on the image
        self.getName(self.image_path)
    def capture_frame(self):
        ret, frame = self.cap.read()
        
        if ret:
            data = frame
            cv2.imwrite("local_photo.jpeg", data)
            self.image_path = "local_photo.jpeg"
            self.btn_predict["state"] = tk.ACTIVE
            
    def open_camera(self):

            self.cap = cv2.VideoCapture(0)
            self.update_frame()

    def update_frame(self):
        # Read a frame from the video capture
        ret, frame = self.cap.read()
        if ret:
            cv2.imwrite("tmp.jpeg", frame)
            # Convert the image from BGR to RGB
            self.photo = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.photo = cv2.resize(self.photo, (600, 400))
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.photo))
            
            # Display the image on the canvas
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        # Call this method again after a delay to update the frame
            try:
                self.getName("tmp.jpeg",live=True)
                # put label name on image bottom left
                self.canvas.create_text(10, 390, anchor='sw', text=self.lable_name,fill='white',font=("Purisa", 20))
            except:
                pass   
        
        self.window.after(10, self.update_frame)

    def getName(self,path : str,live = False):

        # Define the image transformation
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Load and transform the image
        image_path = path  # replace with your image path
        image = Image.open(image_path)
        # what unsqueeze does is that it adds a dimension to the tensor at the specified position
        image = transform(image).unsqueeze(0)
        
        # Perform inference
        with torch.no_grad():
            outputs = self.model(image)

        # Get the predicted class
        _, predicted = torch.max(outputs, 1)

        # get the top 5 predictions with probabilities
        probabilities = F.softmax(outputs, dim=1)
        
        listOfProbabilities, listOfPrediction = torch.topk(probabilities, 5)
        
        
        
        
        

        

        with open('imagenet.json') as json_file:
            labels_dict = json.load(json_file)
        img_final = cv2.imread(self.image_path)
        self.lable_name = labels_dict[str(predicted.item())]
        # print(f'the class name is {self.lable_name}')
        if live != True:
             # clearing the terminal
            os.system('clear')
            for i  in enumerate(listOfPrediction[0]):
                percent = "{:.2f}".format(listOfProbabilities[0][i[0]].item()*100)
                print(f"the class name is {labels_dict[str(i[1].item())]} with probability {percent} %" )
                
                cv2.putText(img_final, f"{labels_dict[str(i[1].item())]} with probability {percent} %", (10,10 + (i[0] + 1) * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            else:
                print('\n')
            
            cv2.putText(img_final, self.lable_name, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            cv2.imshow("Predicted",img_final)
            # print(listOfProbabilities)
            # print(listOfPrediction)
            self.send_status.set(self.lable_name)
            



        






root = tk.Tk()
root.geometry('400x200')
root.resizable(False, False)
root.title("A App")

root.withdraw()
video_player_window = tk.Toplevel(root)
video_player_window.title("Video Player")

video_player_app = VideoPlayerApp(video_player_window, "Video Player")
