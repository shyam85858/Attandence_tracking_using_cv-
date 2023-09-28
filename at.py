# Import necessary libraries
import cv2
import numpy as np
import face_recognition as face_rec
import os
import pyttsx3 as textSpeach
from datetime import datetime
import login
import logouttime
import logout

# Initialize a text-to-speech engine
engine = textSpeach.init()

# Define a function to resize an image
def resize(img, size):
    width = int(img.shape[1] * size)
    height = int(img.shape[0] * size)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)

# Define the path to the directory containing student images
path = 'stu'
studentImg = []
studentName = []
myList = os.listdir(path)

# Load student images and names from the directory
for cl in myList:
    curimg = cv2.imread(f'{path}/{cl}')
    studentImg.append(curimg)
    studentName.append(os.path.splitext(cl)[0])

# Define a function to find face encodings for a list of images
def findEncoding(images):
    imgEncodings = []
    for img in images:
        img = resize(img, 0.50)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeimg = face_rec.face_encodings(img)[0]
        imgEncodings.append(encodeimg)
    return imgEncodings

# Initialize empty lists to store attendance information
n1 = []  # Names
p1 = []  # Student IDs
t1 = []  # Timestamps
d1 = []  # Dates

# Function to mark attendance
def MarkAttendence(name, pn):
    now = datetime.now()
    timestr = now.strftime('%H:%M:%S')
    datestr = now.strftime('%d-%m-%y')
    if name not in n1:
        n1.append(name)
        p1.append('20R21A0' + pn.upper())
        t1.append(timestr)
        d1.append(datestr)

# Call the login function
login.login()

# Get the logout time from another function
p, q = logouttime.logouttime()

# Find face encodings for known students
EncodeList = findEncoding(studentImg)

# Initialize the webcam video capture
vid = cv2.VideoCapture(0)

while True:
    # Read a frame from the video
    success, frame = vid.read()
    Smaller_frames = cv2.resize(frame, (0, 0), None, 0.25, 0.25)

    # Detect faces in the frame
    facesInFrame = face_rec.face_locations(Smaller_frames)
    encodeFacesInFrame = face_rec.face_encodings(Smaller_frames, facesInFrame)

    for encodeFace, faceloc in zip(encodeFacesInFrame, facesInFrame):
        # Compare the detected face with known student faces
        matches = face_rec.compare_faces(EncodeList, encodeFace)
        facedis = face_rec.face_distance(EncodeList, encodeFace)
        matchIndex = np.argmin(facedis)

        if matches[matchIndex]:
            name = studentName[matchIndex][3:].upper()
            pn = studentName[matchIndex][0:3]
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(frame, (x1, y2 - 25), (x2, y2), (0, 0, 255), cv2.FILLED)
            # Display the student's name
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            # Mark the attendance
            MarkAttendence(name, pn)

    # Display the video feed
    cv2.imshow('video', frame)
    cv2.waitKey(1)

    # Call the logout function
    logout.logout(p, q, n1, p1, t1, d1)
