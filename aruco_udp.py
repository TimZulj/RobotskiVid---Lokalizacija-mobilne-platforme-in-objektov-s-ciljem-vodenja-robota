# Tom Turk, Tim Žulj - Robotski vid
# Zaznavanje ArUco markerjev

import numpy as np
import cv2
import cv2.aruco as aruco

import socket
import struct
import sys


def main():

    client_ip = "192.168.65.46" # IP na katerega pošljemo podatke
    port = 25000
    marker_size = 0.2 # Velikost ArUco markerjev je 20 cm
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP
     
    calibrationFile = "logitech_c920yml.yml"
    calibrationParams = cv2.FileStorage(calibrationFile, cv2.FILE_STORAGE_READ)
    camera_matrix = calibrationParams.getNode("camera_matrix").mat()
    dist_coeffs = calibrationParams.getNode("distortion_coefficients").mat()
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    parameters =  aruco.DetectorParameters_create()
    print(parameters)
    parameters.adaptiveThreshWinSizeMin =  5
    parameters.adaptiveThreshWinSizeMax = 23
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    cap = cv2.VideoCapture(0)
     
    while(True):
        # Zajemanje slike
    
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Pretvorba v sivinsko sliko
    
        # Zaznavanje oglišč in identifikacija markerjev
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
        vals = np.zeros((7,3), dtype=np.float32 )
            
        if np.all(ids != None):   
    
            rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)
    
            for i in range(0, ids.size):
                aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.1)  # Izris koordinatnih osi
                vals[0:3,i] = rvecs[i] # Shranjevanje rotacijskih vektorjev
                vals[3:6,i] = tvecs[i] # Shranjevanje translacijskih vektorjev
                vals[6,i] = ids[i] # Shranjevanje identifikacije markerja

            # Urejanje podatkov za pošiljanje prek UDP v Simulink
            vals1 = vals[0:7, 0]
            vals_woke = np.hstack((vals1, vals[0:7, 1]))
            vals_woke = np.hstack((vals_woke, vals[0:7, 2]))
            aruco.drawDetectedMarkers(frame, corners) #Izris markerja

            # Izris identifikacije
            strg = ''
            for i in range(0, ids.size):
                strg += str(ids[i][0])+', '

            cv2.putText(frame, "Id: " + strg, (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)

        else:
            # Izris v primeru, če markerji niso zaznani
            cv2.putText(frame, "No Ids", (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)
            
        vals = vals_woke.reshape(21,1)
        bin_vals = struct.pack('<21f',*vals)
        sock.sendto(bin_vals, (client_ip, port)) # Pošiljanje podatkov na izbrani port
        sys.stdout.flush()            
        
        # Prikaz končne slike
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
     
    # Na koncu nehamo zajemati sliko
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
