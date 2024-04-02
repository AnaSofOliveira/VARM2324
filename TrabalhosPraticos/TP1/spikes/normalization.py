import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

print(os.getcwd())
path = 'TrabalhosPraticos/TP1/images/originais/'
filename = 'ao1.jpg'

# Load image
imagem_original = cv2.imread(path + filename)

# Convert to RGB
imagem_rgb = cv2.cvtColor(imagem_original, cv2.COLOR_BGR2RGB)

# Convert to Grayscale
imagem_cinza = cv2.cvtColor(imagem_original, cv2.COLOR_BGR2GRAY)

# Present the original image using matplotlib
plt.imshow(imagem_rgb)
plt.title('Imagem Original')
plt.axis('off')

# Present the gray image using matplotlib
plt.imshow(imagem_cinza, cmap='gray')
plt.title('Imagem Cinza')
plt.axis('off')

# Load Haar cascade model to face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

# Detect faces
faces = face_cascade.detectMultiScale(imagem_cinza, scaleFactor=1.1, minNeighbors=4)

# Identify the biggest face found
face = faces[np.argmax([w*h for _, _, w, h in faces])]

# Create an image with the face drawn
imagem_marcada = imagem_rgb.copy()
x_face, y_face, w_face, h_face = face
imagem_marcada = cv2.rectangle(imagem_marcada, (x_face, y_face), (x_face+w_face, y_face+h_face), (0, 255, 0), 3)

# Present the original image with marked face using matplotlib
plt.imshow(imagem_marcada)
plt.title('Imagem com Face Marcada')
plt.axis('off')

# Crop the face from the gray image
face_region = imagem_cinza[y_face:y_face+h_face, x_face:x_face+w_face]


# Load Haar cascade model to eyes detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye_tree_eyeglasses.xml')

# Detect eyes
eyes = eye_cascade.detectMultiScale(face_region, scaleFactor=1.1, minNeighbors=4)

print("Eyes: \n", eyes)

# Draw the eyes on marked image
for x, y, w, h in eyes:
    eye_radious = int(round((w + h) * 0.25))
    eye_center = (x_face + x + w//2, y_face+ y + h//2)
    imagem_marcada = cv2.circle(imagem_marcada, eye_center, eye_radious, (255, 0, 0), 3)

# Present the original image with marked face and eyes using matplotlib
plt.imshow(imagem_marcada)
plt.title('Imagem com Face e Olhos Marcados')
plt.axis('off')


# Order eyes by x position
eyes = sorted(eyes, key=lambda pos: pos[0])
print("Ordered eyes: \n", eyes)

# Get left eye coordinates
x_left_eye, y_left_eye, w_left_eye, h_left_eye = eyes[0]
eye_center_left = (x_face + x_left_eye + w_left_eye//2, y_face+ y_left_eye + h_left_eye//2)

# Get right eye coordinates
x_right_eye, y_right_eye, w_right_eye, h_right_eye = eyes[1]
eye_center_right = (x_face + x_right_eye + w_right_eye//2, y_face+ y_right_eye + h_right_eye//2)

# Calculate the angle between the eyes
eyes_angle = np.degrees(np.arctan2(-(eye_center_right[1]-eye_center_left[1]), eye_center_right[0]-eye_center_left[0]))
print("Eyes angle: ", eyes_angle)

# Rotate original gray image to align eyes horizontally in gray image coordinates
center = (int(eye_center_left[0]), int(eye_center_left[1]))
M = cv2.getRotationMatrix2D(center, 360-eyes_angle, 1.0)
face_region = cv2.warpAffine(imagem_cinza, M, imagem_cinza.shape[:2])

# Present the rotated face using matplotlib
plt.imshow(face_region, cmap='gray')
plt.title('Imagem com Face Rotacionada')
plt.axis('off')

face_region = face_region[y_face:y_face+h_face, x_face:x_face+w_face]

# Present the rotated face using matplotlib
plt.imshow(face_region, cmap='gray')
plt.title('Imagem com Face Recortada e Rotacionada')
plt.axis('off')

# Resize the face region so that left eye is placed on column 16 and right eye is placed on column 31
left_eye_column = 16
right_eye_column = 31

# Calculate the desired distance between the eyes
desired_eyes_distance = right_eye_column - left_eye_column

# Calculate the distance between the eyes
eyes_distance = np.sqrt((eye_center_right[0]-eye_center_left[0])**2 + (eye_center_right[1]-eye_center_left[1])**2)
print("Eyes distance: ", eyes_distance)

# Calculate the new distance between the eyes
new_eyes_distance = 15
print("New eyes distance: ", new_eyes_distance)

# Calculate the new size of the image
new_size = (int(imagem_cinza.shape[1]*new_eyes_distance/eyes_distance), int(imagem_cinza.shape[0]*56/imagem_cinza.shape[0]))
print("New size: ", new_size)

# Resize the image
resized_face_region = cv2.resize(imagem_cinza, new_size, interpolation = cv2.INTER_AREA)


new_x_left_eye = int(round((x_left_eye + x_face)*new_eyes_distance/eyes_distance)) 
new_y_left_eye = int(round((y_left_eye + y_face)*new_eyes_distance/eyes_distance))
new_x_right_eye = new_x_left_eye
new_y_right_eye = new_y_left_eye+15

print("NEW LEFT EYE POS:", new_x_left_eye, new_y_left_eye)
print("NEW RIGHT EYE POS:", new_x_right_eye, new_y_right_eye)


cv2.imwrite('TrabalhosPraticos/TP1/images/normalizadas/first_' + filename, resized_face_region)


# Cut image in order to have the center of the right eye in line 24 and column 16, and the center of the left eye in line 24 and column 31
left_eye_column = 16
right_eye_column = 31
left_eye_line = 24
right_eye_line = 24

# Calculate the new coordinates of the eyes
#new_x_left_eye = int(round(x_left_eye*new_eyes_distance/eyes_distance))
#new_y_left_eye = int(round(y_left_eye*new_eyes_distance/eyes_distance))
#new_x_right_eye = int(round(x_right_eye*new_eyes_distance/eyes_distance))
#new_y_right_eye = int(round(y_right_eye*new_eyes_distance/eyes_distance))

# Calculate the new coordinates of the face region
new_x_face = int(round(x_face*new_eyes_distance/eyes_distance))
new_y_face = int(round(y_face*new_eyes_distance/eyes_distance))
new_w_face = int(round(w_face*new_eyes_distance/eyes_distance))
new_h_face = int(round(h_face*new_eyes_distance/eyes_distance))

print("Right Eye: ", new_x_right_eye, new_y_right_eye)
print("Left Eye: ", new_x_left_eye, new_y_left_eye)

print("X:" , new_x_right_eye-16)
print("Y:" , new_y_right_eye-24)
print("W: ", new_x_right_eye-16+46)
print("H: ", new_y_right_eye-24+56)

# Crop the face region
resized_face_region = resized_face_region[new_x_right_eye-16:new_x_right_eye-16+46, new_y_right_eye-24:new_y_right_eye-24+56]

# Resize the face region
# resized_face_region = cv2.resize(face_region, (46, 56), fx=scale_factor, fy=face_region.shape[1]/56, interpolation=cv2.INTER_CUBIC)

# Present the resized face using matplotlib
plt.figure()
plt.imshow(resized_face_region, cmap='gray')
plt.title('Imagem com Face Redimensionada')
plt.axis('off')
plt.show()


cv2.imshow("Imagem 1", face_region)
cv2.imshow("Imagem 2", resized_face_region)
cv2.waitKey(0)
cv2.destroyAllWindows()


cv2.imwrite('TrabalhosPraticos/TP1/images/normalizadas/' + filename, resized_face_region)



    

