import cv2
import numpy
import os

face_file = 'haarcascade_frontalface_default.xml'
datasets = 'dataset'
sub_data = 'sub'

path = os.path.join(datasets, sub_data)
if not os.path.isdir(path):
    os.mkdir(path)

width, height = (500, 500)
face_cascade = cv2.CascadeClassifier(face_file)
web_cam = cv2.VideoCapture(0)

count = 1
while count < 50:
    _, im = web_cam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 4)
        face = gray[y:y+h, x:x+w]
        faces_resize = cv2.resize(face, (width, height))
        cv2.imwrite(f'{path}/{count}.png', faces_resize)
    count += 1

    cv2.imshow('opencv', im)
    key = cv2.waitKey(10)
    cv2.destroyAllWindows()
    if key == 27:
        break