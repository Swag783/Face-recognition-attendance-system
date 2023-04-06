import numpy as np
import cv2
import face_recognition
import os
from datetime import datetime
from datetime import date

def find_encoding(images):
    encodelist = []
    for i in images:
        img = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        print(encode)
        encodelist.append(encode)
    return encodelist


def mark_attendence(name):
    with open('Attendence.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for lines in myDataList:
            entry = lines.split(',')
            nameList.append(entry[0])
        print(entry)
        if name not in nameList:
            today = date.today()
            now = datetime.now()
            dtString = now.strftime("%H:%M:%S")
            f.writelines(f'\n{name}, {dtString},{today}')
            print('Attendence Done')


path = './FaceDetection'
images = []
classNames = []
mylist = os.listdir(path)
print(mylist)
for i in mylist:
    curImg = cv2.imread(f'{path}/{i}')
    images.append(curImg)
    classNames.append(os.path.splitext(i)[0])
print(classNames)

encodeListKnown = find_encoding(images)
print("The Encoding is done")

cap = cv2.VideoCapture(0)  # for vdo capture
while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # This is for making the image size small
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # in the webcam we might have multiple faces for that we have to find the location of the faces
    faceCurFrane = face_recognition.face_locations(imgS)  # find the all the location in our small image
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrane)

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrane):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            # 0, 3, 1, 2  ->> Locations
            x1 = x1 * 4
            x2 = x2 * 4
            y1 = y1 * 4
            y2 = y2 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # rectangle for image
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)  # rectangle for text
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            mark_attendence(name)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.imshow('webcam', img)
    cv2.waitKey(1)

# Allte cooding part is done........The design part is remaining.



