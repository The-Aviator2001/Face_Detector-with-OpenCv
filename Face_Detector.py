import cv2
from random import randrange

#load some pre-trained data on face frontals from opencv (HAAR CASCADE Algorithm)
trained_face_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#choose an image to detect faces
#img= cv2.imread('rdj.jpg')

#Caputure Video
webcam = cv2.VideoCapture(0)
## iterate till frame ends


while True:

    #Read Current FRame
    successful_frame_read, frame= webcam.read();



    #Conversion to Graysacle
    grayscaled_img= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    #Detect Face by using Co-ordinates 
    face_coordinates= trained_face_data.detectMultiScale(grayscaled_img)

    print(face_coordinates)

    #Make a Rectange Around the face
    for (x, y, w, h) in face_coordinates: 
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(256), randrange(256) ,randrange(256)), 2)


    cv2.imshow('Face Detector',frame)
    key = cv2.waitKey(1)

    if key== 81 or key== 113:
      break


## Release  the VideoCapture
webcam.release()



print("Code Completed")