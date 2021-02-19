#!/usr/bin/env python3

import warnings

def warn(*args, **kwargs):
    pass
warnings.warn = warn

import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"

import cv2
from keras.models import load_model
import numpy as np
from pygame import mixer as Deleeper
from time import strftime

print("""\
   || Binit Ghimire and Brinda Subedi - Team #PitamUnga | KU HackFest 2021 ||
 ________    _______  ___       _______   _______    _______    _______   _______   
|"      "\  /"     "||"  |     /"     "| /"     "|  |   __ "\  /"     "| /"      \  
(.  ___  :)(: ______)||  |    (: ______)(: ______)  (. |__) :)(: ______)|:        | 
|: \   ) || \/    |  |:  |     \/    |   \/    |    |:  ____/  \/    |  |_____/   ) 
(| (___\ || // ___)_  \  |___  // ___)_  // ___)_   (|  /      // ___)_  //      /  
|:       :)(:      "|( \_|:  \(:      "|(:      "| /|__/ \    (:      "||:  __   \  
(________/  \_______) \_______)\_______) \_______)(_______)    \_______)|__|  \___) 

        || Detecting Sleepers | Sleeping Driver Detection and Warning ||
        
                            Press "x" to EXIT!
""")
 
def main():
    # Initializing Deleeper!
    Deleeper.init()
    
    # Defining the alert sound!
    sound = Deleeper.Sound("music/alert.wav")
    
    # Defining the Haar Cascade Classifier Algorithm path!
    face = cv2.CascadeClassifier("HaarCascadeClassifiers\haarcascade_frontalface_alt.xml")
    leftEye = cv2.CascadeClassifier("HaarCascadeClassifiers\haarcascade_lefteye_2splits.xml")
    rightEye = cv2.CascadeClassifier("HaarCascadeClassifiers\haarcascade_righteye_2splits.xml")
    
    # Defining the labels!
    label=["Closed!","Opened!"]
    
    # Getting current system path!
    path = os.getcwd()
    
    # Loading the Models!
    model = load_model("models/model.h5")
    
    # Starting the user's video capture device!
    capture = cv2.VideoCapture(0)
    
    # Defining the Deleeper core components and values!
    font = cv2.FONT_HERSHEY_TRIPLEX

    count = 0
    points = 0

    thick = 2

    rightPrediction = [99]
    leftPrediction = [99]
    
    while(True):
        ret,frame = capture.read()
        height,width = frame.shape[:2]
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
        faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
        leftEyeDetect = leftEye.detectMultiScale(gray)
        rightEyeDetect =  rightEye.detectMultiScale(gray)
        
        # Team #PitamUnga | KU HackFest 2021!
        cv2.putText(frame, "#PitamUnga | KU HackFest 2021", (36, 36), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Rectangles!
        cv2.rectangle(frame, (8, height-50), (164, height-8), (36,18,230), thickness=cv2.FILLED)
        cv2.rectangle(frame, (width-216, height-50), (width, height-8), (36,18,230), thickness=cv2.FILLED)
        
        # Face Recognition!
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (100,100,100), 1)

        # Right Eye Recognition and Right Eye Status Prediction!
        for (x,y,w,h) in rightEyeDetect:
            rightEyeVal=frame[y:y+h,x:x+w]
            count=count+1
            rightEyeVal = cv2.cvtColor(rightEyeVal, cv2.COLOR_BGR2GRAY)
            rightEyeVal = cv2.resize(rightEyeVal, (24, 24))
            rightEyeVal = rightEyeVal/255
            rightEyeVal =  rightEyeVal.reshape(24, 24, -1)
            rightEyeVal = np.expand_dims(rightEyeVal, axis=0)
            rightPrediction = model.predict_classes(rightEyeVal)
            if(rightPrediction[0]==1):
                label = "Opened!" 
            if(rightPrediction[0]==0):
                label = "Closed!"
            break

        # Left Eye Recognition and Left Eye Status Prediction!
        for (x,y,w,h) in leftEyeDetect:
            leftEyeVal = frame[y:y+h,x:x+w]
            count=count+1
            leftEyeVal = cv2.cvtColor(leftEyeVal,cv2.COLOR_BGR2GRAY)  
            leftEyeVal = cv2.resize(leftEyeVal,(24,24))
            leftEyeVal = leftEyeVal/255
            leftEyeVal =leftEyeVal.reshape(24,24,-1)
            leftEyeVal = np.expand_dims(leftEyeVal,axis=0)
            leftPrediction = model.predict_classes(leftEyeVal)
            if(leftPrediction[0]==1):
                label = "Opened!"   
            if(leftPrediction[0]==0):
                label = "Closed!"
            break
        
        # Checking whether the eyes are Opened or Closed!
        if(rightPrediction[0]==0 and leftPrediction[0]==0):
            if points!=99:
                points = points+1
            cv2.putText(frame, "Closed!", (20, height-20), font, 1,(255, 255, 255), 1, cv2.LINE_AA)
        else:
            points=points-1
            cv2.putText(frame, "Opened!", (20, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Points for the Eye Status!
        if(points<0):
            points=0
        cv2.putText(frame, "Points: "+str(points), (width-200, height-20), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Playing sound when the eyes are closed, and the points become higher than 15!
        if(points>15):
            if not os.path.exists(os.path.join(path, "images", str(strftime("%a, %d %b, %Y")))):
                os.makedirs(os.path.join(path, "images", str(strftime("%a, %d %b, %Y"))))
            cv2.imwrite(os.path.join(path, "images", str(strftime("%a, %d %b, %Y")), str(strftime("%H-%M-%S")) + ".jpg"), frame)
            try:
                sound.play()
            except:
                pass
            if(thick<16):
                thick = thick + 2
            else:
                thick = thick - 2
                if(thick<2):
                    thick = 2
            cv2.rectangle(frame, (0,0), (width, height), (0, 0, 255), thick)

        # Deleeper Window Status!
        cv2.imshow("Deleeper", frame)
        if cv2.waitKey(1) & 0xFF == ord("x"):
            print("\t\t    || Thank You for using Deleeper! ||\n")
            break

    # Releasing the Device Camera and Closing all Deleeper windows!
    capture.release()
    cv2.destroyAllWindows()

# Starting Deleeper!
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\t\t    || Thank You for using Deleeper! ||\n")
        os._exit(1)