import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot

cap = cv2.VideoCapture("video1.mp4")
detector = FaceMeshDetector()
plotY = LivePlot(540, 360, [10,60])

id_list = [22,23,24,26,110,157,158,159,160,161,130,243]
color = (0,0,255)
ratioList = []
blinkCounter = 0
counter = 0
while  True:
    ret, frame = cap.read()
    if not ret:break
    frame, faces = detector.findFaceMesh(frame, draw=False)
    
    if faces:
        face = faces[0]
        for id in id_list:
            cv2.circle(frame, face[id], 5, color, cv2.FILLED) 

        leftUp = face[159]
        leftDown = face[23]
        leftLeft = face[130]
        leftRight = face[243]

        lengthVer, _ = detector.findDistance(leftUp, leftDown)
        lengthHor, _ = detector.findDistance(leftLeft, leftRight)

        cv2.line(frame, leftUp, leftDown, (0,255,0),3)
        cv2.line(frame, leftLeft, leftRight,(0,255,0),3)

        ratio = int((lengthVer/lengthHor) * 100)
        ratioList.append(ratio)
        if len(ratioList) > 3:
            ratioList.pop(0)

        ratioAvg = sum(ratioList) / len(ratioList)
        print(ratioAvg)

        if ratioAvg < 35 and counter == 0:
            blinkCounter += 1 
            color = (0,255,0)
            counter += 1
        if counter != 0:
            counter += 1
            if counter > 10:
                counter = 0
                color = (0,0,255)

        cvzone.putTextRect(frame, f"Blink Count: {blinkCounter}", (50,100), colorR=color)

    imgPlot = plotY.update(ratioAvg,color)
    frame = cv2.resize(frame, (640,360))
    imgStack = cvzone.stackImages([frame,imgPlot], 2, 1)

    cv2.imshow("frame", imgStack)
    if cv2.waitKey(25) & 0xFF == ord("q"):break

cap.release()
cv2.destroyAllWindows()
