import cv2 as cv

def detect(img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    face_detect = cv.CascadeClassifier('C:/Users/A/AppData/Local/Programs/Python/Python39/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
    face = face_detect.detectMultiScale(gray)
    for x,y,w,h in face:
        cv.rectangle(img,(x,y),(x+w,y+h),color=(0,0,255),thickness=5)
    cv.imshow('result',img)

# video = cv.VideoCapture('csgo.mp4')
video = cv.VideoCapture(0)

while True:
    flag,frame = video.read()
    if not flag:
        break
    detect(frame)
    if ord('q') == cv.waitKey(1):
        break
cv.destroyAllWindows()
video.release()