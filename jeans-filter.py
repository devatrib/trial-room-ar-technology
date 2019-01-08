import os
import subprocess
import cv2

face_csc = cv2.CascadeClassifier('C:/projects/opencv-python/opencv/data/haarcascades/haarcascade_lowerbody.xml')
cam = cv2.VideoCapture(0)
hat = cv2.imread('mid_jeans.jpg')

def put_hat(hat,fc,x,y,w,h):
    
    face_width = w
    face_height = h
    
    hat_width = face_width
    hat_height = int(face_height)+10
    
    hat = cv2.resize(hat,(hat_width,hat_height))
    
    for i in range(hat_height):
        for j in range(hat_width):
            for k in range(3):
                if hat[i][j][k]<235:
                    fc[y+i-int(0.25*face_height)][x+j][k] = hat[i][j][k]
    return fc

while(True):
    tf, img = cam.read()       
       
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    faces = face_csc.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40,40)
    )
    
    for (x, y, w, h) in faces:
        cv2.rectangle(gray, (x,y), (x+w, y+h), (0,0,0), 5)
        img = put_hat(hat,img,x,y,w,h)
        
    cv2.imshow('video', img)
    key = cv2.waitKey(1)
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()
