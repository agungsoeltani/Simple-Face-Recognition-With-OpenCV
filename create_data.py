import cv2
import os
import tkinter as tk
from tkinter import simpledialog

haar_file = 'haarcascade_frontalface_default.xml'

# Folder where all the faces data will be stored
datasets = 'datasets'

# Define the size of images
(width, height) = (130, 100)

# Load the Haar cascade file
face_cascade = cv2.CascadeClassifier(haar_file)

# Open the webcam
video = cv2.VideoCapture(0)

# Function to get user input via a dialog box
def get_name():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    name = simpledialog.askstring("Input", "Masukan nama saudara (Atau ketik 'exit' untuk keluar):")
    root.destroy()
    return name

# Collect data for multiple people
while True:
    # Ask for the name of the person
    name = get_name()
    if name is None or name.lower() == 'exit':
        break

    # Path to store the dataset
    path = os.path.join(datasets, name)
    if not os.path.isdir(path):
        os.makedirs(path)

    print(f"Collecting data for {name}. Please look at the camera.")

    count = 1
    while count <= 30:
        ret, frame = video.read()
        if not ret:
            break  # Exit the loop if there are no more frames

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (width, height))
            cv2.imwrite(f'{path}/{count}.png', face_resize)
            count += 1
            if count > 30:
                break

        cv2.imshow('OpenCV', frame)
        key = cv2.waitKey(10)
        if key == 27:  # Press 'ESC' to exit early
            break

    print(f"Collected {count-1} images for {name}.")

video.release()
cv2.destroyAllWindows()
