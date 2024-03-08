import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from math import degrees, atan2

# Percorrer a pasta das imagens originais
# Identificar caras e olhos
# Desenhar um retângulo à volta das caras e circulo no centro dos olhos
# Guardar as imagens com faces cortadas 

# Load the pre-trained model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye_tree_eyeglasses.xml')

for root, dirs, files in os.walk("TrabalhosPraticos/TP1/images/originais/"):
    for file in files:
        filename, extention = file.split(".") # os.path.split(files, ".")
        print(filename, extention)

        # Load the image
        img = cv2.imread("TrabalhosPraticos/TP1/images/originais/" + file)

        # Resize the image
        #img = cv2.resize(img, (int(img.shape[1]*0.8), int(img.shape[0]*0.8)))

        # Convert into grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

        # Identify the biggest face found
        face = faces[np.argmax([w*h for x, y, w, h in faces])]

        (x_face, y_face, w_face, h_face) = face

        # Crop the face
        face_region = gray[y_face:y_face+h_face, x_face:x_face+w_face]
        cv2.imwrite("TrabalhosPraticos/TP1/images/processed/cropped_faces/" + filename + "_preNomalization." + extention, face_region)
        
        # Detect eyes
        eyes = eye_cascade.detectMultiScale(face_region, scaleFactor=1.1, minNeighbors=4)
        eyes_info = []
        for x, y, w, h in eyes:
            if x is not None: 
                eye_radious = int(round((w + h) * 0.25))
                eye_center = (x_face + x + w//2, y_face+ y + h//2)

                eyes_info.append((x, y, w, h, eye_center, eye_radious))
                #if eye_radious > 100:   
                img = cv2.circle(img, eye_center, eye_radious, (255, 0, 0), 3)


        img = cv2.rectangle(img, (x_face, y_face), (x_face+w_face, y_face+h_face), (0, 255, 0), 3)
        cv2.imwrite("TrabalhosPraticos/TP1/images/processed/" + file, img)

        # Order eyes by y position
        eyes_info = sorted(eyes_info, key=lambda pos: pos[0])
        print("* Sorted Eyes: \n", eyes)
        eyes_angle = degrees(atan2(-(eyes_info[1][1]-eyes_info[0][1]), eyes_info[1][0]-eyes_info[0][0]))
        print("* Eyes angle: ", eyes_angle)
        print("* Eyes info: ", eyes_info[0][4])

        # Rotate the image
        center = ((int(eyes_info[0][4][0]), int(eyes_info[0][4][1]))) # w//2, h//2 

        print("* Image center: ", center, type(center), type(center[0]))
        M = cv2.getRotationMatrix2D(center, 360-eyes_angle, 1.0)
        face_region = cv2.warpAffine(gray, M, gray.shape[:2])
        face_region = face_region[y_face:y_face+h_face, x_face:x_face+w_face]
        cv2.imwrite("TrabalhosPraticos/TP1/images/processed/cropped_faces/" + filename + "_normalized." + extention, face_region)



        # Resize the image
        '''(...) each face is represented by a monochrome image (256 levels) with 56 rows and
        46 columns, with both eyes, right and left, perfectly aligned horizontally, and located in
        line 24, columns 16 and 31, respectively.'''

        normalized_face = np.ones((56, 46), dtype=np.uint8)*255.
        eyes_dist = eyes_info[1][4][0] - eyes_info[0][4][0]
        print("* Eyes distance: ", eyes_dist)
        resized = cv2.resize(face_region, fx=16/eyes_dist, fy=16/eyes_dist, dsize=(46, 56), interpolation=cv2.INTER_AREA)
        
        cv2.imwrite("TrabalhosPraticos/TP1/images/processed/cropped_faces/" + filename + "_normalized." + extention, resized)

        cv2.imshow('Resized', resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        # Show the image
        '''cv2.imshow('Image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()'''
    


