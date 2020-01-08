#openCV's - haar cascade algorith
import tkinter as tk
from tkinter import filedialog, font
from tkinter.constants import CENTER

import cv2 as cv
from PIL import Image, ImageTk


def open_a_file():

    video_formats = ".mp4 .mkv .mov"
    global filename
    filename = tk.filedialog.askopenfilename(initialdir="/",title="Select File",
                filetypes= [("Video", video_formats)])
    
    labelfont = tk.font.Font(family='Helvetica', size=12, weight='bold')
    label = tk.Label(text="Location : "+filename, fg=my_foreground_colour, bg=my_background_colour,
                    font=labelfont, wraplength=400, justify="left")
    label.place(relx=0.5,rely=0.15,anchor=CENTER)

def process():
    if(filename!=None):
        print("Filename : ",filename)
        capture = cv.VideoCapture(filename)
        haar_cascase = cv.CascadeClassifier ('haar_face.xml')
        while True : 
            
            isTrue,frame = capture.read()
            if(isTrue==False):
                break
            gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
            faces_rect = haar_cascase.detectMultiScale(gray,1.1,3)
            for (x,y,w,h) in faces_rect:
                cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),thickness=2)
            cv.imshow('Detected faces in Video',frame)
            if cv.waitKey(1) & 0xFF==ord('d'):
                break
            if cv.getWindowProperty('Detected faces in Video', cv.WND_PROP_VISIBLE) <1:
                break
        capture.release()
        cv.destroyAllWindows()

def process_webcam():

    capture = cv.VideoCapture(0)
    haar_cascase = cv.CascadeClassifier ('haar_face.xml')

    while True:
        isTrue,frame = capture.read()
        frame = cv.flip(frame,1)
        gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        faces_rect = haar_cascase.detectMultiScale(gray,1.1,3)
        for (x,y,w,h) in faces_rect:
            cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),thickness=2)
        cv.imshow('Detected faces in Video',frame)
        if cv.waitKey(1) & 0xFF==ord('d'):
            break
        if cv.getWindowProperty('Detected faces in Video', cv.WND_PROP_VISIBLE) <1:
            break
        

    capture.release()
    cv.destroyAllWindows()


filename = None
#Initialising Main Window of the application
window = tk.Tk()
window.title("Human Detecting System")
window.geometry("800x800")
window.resizable(False,False)
window.configure(bg="#5cd3ba")



#Initialising font & colours for buttons
buttonFont = font.Font(family='Helvetica', size=16, weight='bold')
my_foreground_colour = "white"
my_background_colour = "#FF6347"



#Initialising open file button
img = Image.open('icons/openfile.jpg')
img = img.resize((50,50), Image.ANTIALIAS)
open_file_icon = ImageTk.PhotoImage(img)
openFile = tk.Button(window,text="Open File",
                    padx=12,pady=5,
                    fg=my_foreground_colour,
                    bg=my_background_colour,
                    image = open_file_icon,compound = tk.LEFT,
                    font=buttonFont,
                    command=open_a_file)
openFile.place(relx=0.5, rely=0.3, anchor=CENTER)



#Initialising process the file button
img = Image.open('icons/processing.jpg')
img = img.resize((50,50), Image.ANTIALIAS)
processing_icon = ImageTk.PhotoImage(img)
process = tk.Button(window,text="Process",
                    padx=12,pady=5,
                    fg=my_foreground_colour,
                    bg=my_background_colour,
                    image = processing_icon,compound = tk.LEFT,
                    font=buttonFont,
                    command=process)
process.place(relx=0.5, rely=0.5, anchor=CENTER)



#Initialising openwebcam button
img = Image.open('icons/webcam.png')
img = img.resize((50,50), Image.ANTIALIAS)
webcam_icon = ImageTk.PhotoImage(img)
process = tk.Button(window,text="Webcam",
                    padx=12,pady=5,
                    fg=my_foreground_colour,
                    bg=my_background_colour,
                    image = webcam_icon,compound = tk.LEFT,
                    font=buttonFont,
                    command=process_webcam)
process.place(relx=0.5, rely=0.7, anchor=CENTER)

window.mainloop()
