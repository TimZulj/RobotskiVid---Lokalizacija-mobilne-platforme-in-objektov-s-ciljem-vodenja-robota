# Tom Turk, Tim Žulj - Robotski vid
# Detekcija objektov

import cv2
import numpy as np
import imutils
import struct
import sys
import socket
'''
# Zajem slike
cam = cv2.VideoCapture(0)
#cv2.namedWindow("test")
img_counter = 0

while True:
    ret, frame = cam.read()
    cv2.imshow("test ", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()
cv2.destroyAllWindows()


###################################################################################################
# Kalibracija kamere
slika = cv2.imread("opencv_frame_0.png")
cv2.imshow("slika",slika)
  
calibrationFile = "logitech_c920yml.yml"
calibrationParams = cv2.FileStorage(calibrationFile, cv2.FILE_STORAGE_READ)
camera_matrix = calibrationParams.getNode("camera_matrix").mat()
dist_coeffs = calibrationParams.getNode("distortion_coefficients").mat()
 
# Velikost slike v pikslih
imageSize = (640, 480)

map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, None, imageSize, 5)
kalibrirana_slika = cv2.remap(slika, map1, map2, interpolation=cv2.INTER_LINEAR)
 
cv2.imshow("Kalibrirana slika", kalibrirana_slika)
img_name = "kalibrirana_best.png".format(0)
cv2.imwrite(img_name, kalibrirana_slika)
cv2.waitKey()
#################################################
'''

# Preberemo shranjeno kalibrirano sliko
slika = cv2.imread("kalibrirana_best.png")


blurred = cv2.GaussianBlur(slika, (5, 5), 0) # Glajenje slike
lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB) # Pretvorba v LAB barvni prostor
slika_hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV) # Pretvorba v HSV

# Zaznavanje zelene obrobe kock za določanje oblike
# Definiramo območje zelene barve v HSV prostoru
lower_green = np.array([40, 30, 30])
upper_green = np.array([80, 255, 255])

# Threshold the HSV image to get only blue colors
mask = cv2.inRange(slika_hsv, lower_green, upper_green)

# Bitwise-AND mask and original image
zelena = cv2.bitwise_and(slika_hsv, slika_hsv, mask=mask)
cv2.imshow('zelena', zelena)

# Zaznavanje modre barve
# Definiramo območje modre barve v HSV prostoru
lower_blue = np.array([105, 100, 10])
upper_blue= np.array([120, 250, 255])

# Upragovljanje slike
mask = cv2.inRange(slika_hsv, lower_blue, upper_blue)

# Definiramo velikost jedra za erozijo in dilacijo
kernel = np.ones((3, 3), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # Erozija, ki ji sledi dilacija
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # Dilacija, ki ji sledi erozija

# Bitni AND maske in slike
modra = cv2.bitwise_and(slika_hsv, slika_hsv, mask=mask)

# Zaznavanje rdeče barve
# Definiramo območje rdeče barve v HSV prostoru
lower_red = np.array([0, 150, 10])
upper_red = np.array([10, 255, 255])

# Upragovljanje slike
mask = cv2.inRange(slika_hsv, lower_red, upper_red)

mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # Erozija, ki ji sledi dilacija
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # Dilacija, ki ji sledi erozija

# Bitni AND maske in slike
rdeca = cv2.bitwise_and(slika_hsv, slika_hsv, mask=mask)

#################################################################


zelena = cv2.cvtColor(zelena, cv2.COLOR_HSV2BGR) # Pretvorba v BGR prostor
zelena_gray = cv2.cvtColor(zelena, cv2.COLOR_BGR2GRAY) # Pretvorba v sivinsko sliko
zelena_gray = cv2.GaussianBlur(zelena_gray, (5, 5), 0) # Glajenje slike

# Upragovljanje slike
thresh_zelena = cv2.threshold(zelena_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

rdeca = cv2.cvtColor(rdeca, cv2.COLOR_HSV2BGR) # Pretvorba v BGR prostor
rdeca_gray = cv2.cvtColor(rdeca, cv2.COLOR_BGR2GRAY) # Pretvorba v sivinsko sliko
rdeca_gray = cv2.GaussianBlur(rdeca_gray, (5, 5), 0) # Glajenje slike

# Upragovljanje slike
thresh_rdeca = cv2.threshold(rdeca_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

modra = cv2.cvtColor(modra, cv2.COLOR_HSV2BGR) # Pretvorba v BGR prostor
modra_gray = cv2.cvtColor(modra, cv2.COLOR_BGR2GRAY) # Pretvorba v sivinsko sliko
modra_gray = cv2.GaussianBlur(modra_gray, (5, 5), 0) # Glajenje slike

# Upragovljanje slike
thresh_modra = cv2.threshold(modra_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

# Bitni OR modrih in rdečih površin
modra_rdeca = cv2.bitwise_or(thresh_modra, thresh_rdeca)

# Bitni AND modrih in rdečih površin ter slike v LAB prostoru
lab = cv2.bitwise_and(lab, lab, mask=modra_rdeca)
lab = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# Prikaz maske zaznanih mordrih in rdečih površin
cv2.imshow('modra + rdeca', modra_rdeca)
cv2.waitKey()

# Najdi konture v sliki zelenih obrob
cnts = cv2.findContours(thresh_zelena.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

from pyimagesearch.shapedetector import ShapeDetector
from pyimagesearch.colorlabeler import ColorLabeler

sd = ShapeDetector()
cl = ColorLabeler()

# Inicializiacija spremenljivk za shranjevanje koordinat centrov
x = []
y = []

for c in cnts:

    # Izračun centrov
    M = cv2.moments(c)
    ratio = 1

    cX = int((M["m10"] / M["m00"]) * ratio)
    cY = int((M["m01"] / M["m00"]) * ratio)

    # Shranjevanje koordinat centrov
    x.append(cX)
    y.append(cY)

    shape = sd.detect(c) # Zaznavanje oblik

    c = c.astype("float")
    c *= ratio
    c = c.astype("int")
    cv2.drawContours(slika, [c], -1, (0, 255, 0), 2) # Izris kontur
    text = "{}{}".format(shape, "")
    cv2.putText(slika, text, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2) # Izpis oblike objekta

# Najdi konture v sliki rdečih in modrih površin
cnts = cv2.findContours(modra_rdeca.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# Inicializiacija spremenljivke za shranjevanje barv
barve = []


for c in cnts:
    
    # Izračun centrov
    M = cv2.moments(c)

    cX = int((M["m10"] / M["m00"]) * ratio)
    cY = int((M["m01"] / M["m00"]) * ratio)

    color = cl.label(lab, c) # Zaznavanje barv

    c = c.astype("float")
    c *= ratio
    c = c.astype("int")
    text = "{}           {}".format("", color)
    cv2.putText(slika, text, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2) # Izpis barv

    if color =='blue': # Kodiranje barv v numerične vrednosti za pošiljanje v Simulink
        color=1
    else:
        color=0

    # Shranjevanje barv v spremenljivko
    barve.append(color)

# Prikaz končne slike
cv2.imshow("Image", slika)
print(barve)
print(x)
print(y)

# Naslov kamor se pošljejo podatki
client_ip = "192.168.65.46"
port = 25005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
x = struct.pack('<2f',*x) # Na prvi port pošljemo dve x koordinati centrov
sock.sendto(x, (client_ip, port))


port = 25010
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

y = struct.pack('<2f',*y) # Na drugi port pošljemo dve y koordinati centrov
sock.sendto(y, (client_ip, port))

port = 25020
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

barve = struct.pack('<2f',*barve) # Na treji port pošljemo informacije o barvi
sock.sendto(barve, (client_ip, port))

# Počakamo do pritiska tipke
cv2.waitKey(0)
sys.stdout.flush()
cv2.waitKey(0)


