import cv2
import numpy
import os

size = 4
face_file = 'haarcascade_frontalface_default.xml'
datasets = 'dataset'
images, labels, names, id = [], [], {}, 0

for subdirs, dirs, files in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        s_path = os.path.join(datasets, subdir)
        for filename in os.listdir(s_path):
            path = s_path + '/' + filename
            label = id
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
        id += 1
width, height = (500, 500)
images, labels = [numpy.array(lis) for lis in [images, labels]]

print(dir(cv2.face))
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)

face_cascade = cv2.CascadeClassifier(face_file)
web_cam = cv2.VideoCapture(0)

while True:
    _, im = web_cam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 4)
        face = gray[y:y + h, x:x + w]
        faces_resize = cv2.resize(face, (width, height))

        prediction = model.predict(faces_resize)
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 4)



        if prediction[1] < 500:
            cv2.putText(im, f'{names[prediction[0]]} | {prediction[1]}', (x-10, y-10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
        else:
            cv2.putText(im, 'not recognized', (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

    cv2.imshow('OpenCV', im)
    key = cv2.waitKey(10)
    if key == 27:
        break

# pip install opencv-contrib-python
