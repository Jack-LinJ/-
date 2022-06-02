import cv2 as cv

import os
import urllib
import urllib.request
from PIL import Image
import numpy as np

def getImageAndLabels(path):
    faceData = []
    labels = []
    imagePaths = [os.path.join(path,f)for f in os.listdir(path)]

    face_detect = cv.CascadeClassifier('C:/Users/A/AppData/Local/Programs/Python/Python39/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')

    for imagePath in imagePaths:
        PIL_image = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_image,'uint8')
        faces = face_detect.detectMultiScale(img_numpy)
        label = int(os.path.split(imagePath)[1].split('.')[0])
        # print(id)
        for x,y,w,h in faces:
            labels.append(label)
            print(id,labels)
            faceData.append(img_numpy[y:y+h,x:x+w])
    print('id=',labels)
    # print('faces=',faceData)
    return faceData,labels

path = './face/'
faces, labels = getImageAndLabels(path)
recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.train(faces,np.array(labels))
recognizer.write('trainer/trainer.yml')