import cv2 as cv
import os
import urllib
import urllib.request


recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
names = [0,0,0,0,0,0,0,0]
def face_detect(img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    face_detect = cv.CascadeClassifier(
        'C:/Users/A/AppData/Local/Programs/Python/Python39/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
    face = face_detect.detectMultiScale(gray)
    for x,y,w,h in face:
        cv.rectangle(img,(x,y),(x+w,y+h),color=(0,0,255),thickness=3)
        cv.circle(img,center=(x+w//2,y+h//2),radius=w//2,color=(0,255,0),thickness=1)

        ids,confidence = recognizer.predict(gray[y:y+h,x:x+w])
        print(ids,confidence)
        if confidence>80:
            global warningtime
            warningtime += 1
            if warningtime>100:
                print('检测不出')
                warningtime = 0
            cv.putText(img,'unknow',(x+10,y-10),cv.FONT_HERSHEY_SIMPLEX,0.75,(0,255,0),1)
        else:
            cv.putText(img,str(names[ids-1]),(x+10,y-10),cv.FONT_HERSHEY_SIMPLEX,0.75,(0,255,0),1)
    cv.imshow('result',img)


def getName():
    path = './face/'
    print(recognizer)
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    for imagePath in imagePaths:
        label = int(os.path.split(imagePath)[1].split('.')[0])
        name = os.path.split(imagePath)[1].split('.')[2]

        names[label-1] = name
    print(names)


warningtime = 0
# cap = cv.VideoCapture('zjm.mp4')
cap = cv.VideoCapture(0)
getName()
while True:
    flag,frame = cap.read()
    if not flag:
        break
    face_detect(frame)
    if ord('q') == cv.waitKey(1):
        break
cv.destroyAllWindows()
cap.release()