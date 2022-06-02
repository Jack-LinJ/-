import cv2 as cv


cap = cv.VideoCapture(0)

flag = 1
num = 1
name = 'linji'
label = 1

while (cap.isOpened()):
    ret_flag,Vshow = cap.read()
    cv.imshow('capture',Vshow)
    k = cv.waitKey(1) & 0xFF
    if k == ord('s'):
        cv.imwrite('./face/'+ str(label)+'.' + str(num)+'.'+name + '.jpg',Vshow)
        print('Save successfully',str(num)+'.'+name+ '.jpg')
        num += 1
    elif k == ord('q'):
        break
cap.release()
cv.destroyAllWindows()