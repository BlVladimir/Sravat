import cv2
import numpy as np


aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    corners, ids, rejected = detector.detectMarkers(frame)
    

    if ids is not None and len(ids) == 4:

        all_corners = np.vstack([c[0] for c in corners])
        

        x, y, w, h = cv2.boundingRect(all_corners)
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        

        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    

    cv2.imshow('ArUco Markers', frame)
    
    cv2.waitKey(1)


cap.release()
cv2.destroyAllWindows()