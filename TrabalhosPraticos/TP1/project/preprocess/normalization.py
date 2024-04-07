import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

class Normalization: 

    def __init__(self, image):

        self.__load_image(image, type(image))

        self.__face_detector = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
        self.__eyes_detector = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye_tree_eyeglasses.xml')

        self.__original_eyes_coords = None
        self.__face, self.__right_eye, self.__left_eye = None, None, None

    def __load_image(self, image, type): 

        if(type == str):
            self.__original_image = cv2.imread(image)
            self.__rgb_image = cv2.cvtColor(self.__original_image, cv2.COLOR_BGR2RGB)
            self.__gray_image = cv2.cvtColor(self.__original_image, cv2.COLOR_BGR2GRAY)
    
        elif(type == np.ndarray):
            self.__original_image = image
            self.__rgb_image = cv2.cvtColor(self.__original_image, cv2.COLOR_BGR2RGB)
            self.__gray_image = cv2.cvtColor(self.__original_image, cv2.COLOR_BGR2GRAY)
    
    def __object_detection(self): 
        objectsFound = None

        if(self.__face is None): 
            objectsFound = self.__face_detector.detectMultiScale(self.__gray_image, scaleFactor=1.1, minNeighbors=4)
            objectsFound = objectsFound[np.argmax([w*h for _, _, w, h in objectsFound])]
            self.__face = objectsFound
            self.__face_image = self.__crop_face(self.__gray_image)
            

        elif(self.__right_eye is None and self.__face is not None):
            objectsFound = self.__eyes_detector.detectMultiScale(self.__face_image, scaleFactor=1.1, minNeighbors=4)

            if(len(objectsFound)==2): 
                objectsFound = sorted(objectsFound, key=lambda pos: pos[0])

                self.__left_eye = {"x": objectsFound[0][0] + self.__face[0] + objectsFound[0][2]//2,
                                    "y": objectsFound[0][1] + self.__face[1] + objectsFound[0][3]//2,
                                    "radius": objectsFound[0][2]//2}
                
                self.__right_eye = {"x": objectsFound[1][0] + self.__face[0] + objectsFound[1][2]//2,
                                    "y": objectsFound[1][1] + self.__face[1] + objectsFound[1][3]//2,
                                    "radius": objectsFound[1][2]//2}
                self.__eyes_angle = np.degrees(np.arctan2(-(self.__right_eye["y"]-self.__left_eye["y"]), self.__right_eye["x"]-self.__left_eye["x"]))
                self.__eyes_distance = np.sqrt((self.__right_eye["x"]-self.__left_eye["x"])**2 + (self.__right_eye["y"]-self.__left_eye["y"])**2)
            
                self.__original_eyes_coords = (self.__left_eye, self.__right_eye)

        return objectsFound
    
    def __mark_object_in_image(self, image, object='face'):
        marked_image = image.copy()
        if(object == 'face'):
            x_face, y_face, w_face, h_face = self.__face
            marked_image = cv2.rectangle(marked_image, (x_face, y_face), (x_face+w_face, y_face+h_face), (0, 255, 0), 3)

        elif(object=='eyes'): 
            marked_image = cv2.circle(marked_image, (self.__right_eye["x"], self.__right_eye["y"]) , self.__right_eye["radius"], (255, 0, 0), 3)
            marked_image = cv2.circle(marked_image, (self.__left_eye["x"], self.__left_eye["y"]) , self.__left_eye["radius"], (255, 0, 0), 3)

        return marked_image
    
    def __crop_face(self, image): 
        return image[self.__face[1]:self.__face[1]+self.__face[3], self.__face[0]:self.__face[0]+self.__face[2]]
    

    def __rotate_image(self):

        if(self.__right_eye is not None):
            center = (int(self.__right_eye["x"]), int(self.__right_eye["y"]))
            M = cv2.getRotationMatrix2D(center, 360-self.__eyes_angle, 1.0)
            self.__rotated_image = cv2.warpAffine(self.__gray_image, M, (self.__gray_image.shape[1],self.__gray_image.shape[0]))
            
        else: 
            print("No feature points detected to support rotation. ")
        

    def __resize_image(self): 
        if(self.__right_eye is not None and self.__left_eye is not None):

            desired_eyes_distance = 15
            self.__resize_factor = desired_eyes_distance/self.__eyes_distance

            self.__resized_image = cv2.resize(self.__gray_image, None, fx=self.__resize_factor, fy=self.__resize_factor, interpolation = cv2.INTER_AREA)

        else: 
            print("No feature points detected to support resizing. ")

    def __crop_norm_face(self):
        self.__left_eye = {"x": self.__left_eye["x"]*self.__resize_factor, 
                           "y": self.__left_eye["y"]*self.__resize_factor, 
                           "radius": self.__left_eye["radius"]*self.__resize_factor}
        
        self.__right_eye = {"x": self.__right_eye["x"]*self.__resize_factor,
                            "y": self.__right_eye["y"]*self.__resize_factor,
                            "radius": self.__right_eye["radius"]*self.__resize_factor}

        LEPosX = 16
        LEPosY = 24
        imgWidth = 46
        imgHeight = 56

        self.__norm_face = self.__resized_image[int(self.__left_eye["y"]-LEPosY):int(self.__left_eye["y"]-LEPosY+imgHeight), 
                                                int(self.__left_eye["x"]-LEPosX):int(self.__left_eye["x"]-LEPosX+imgWidth)]

    def save_image(self, image, name): 
        cv2.imwrite(name, image)

    def __show_image(self, image, title, cmap='viridis'): 
        plt.imshow(image, cmap=cmap)
        plt.title(title)
        plt.axis('off')
        plt.show()

    def get_original_face_and_eyes_coords(self):
        return self.__face, self.__original_eyes_coords

    def normalize(self, show_images = False):

        try:
            face = self.__object_detection()
            marked_image = self.__mark_object_in_image(self.__rgb_image)
            eyes = self.__object_detection()
            marked_image = self.__mark_object_in_image(marked_image, object='eyes')

            self.__rotate_image()

            self.__resize_image()

            self.__crop_norm_face()

            if show_images: 
                self.__show_image(self.__rgb_image, "Original Image")
                self.__show_image(self.__gray_image, "Gray Image", cmap='gray')
                self.__show_image(marked_image, "Marked Image")
                self.__show_image(self.__rotated_image, "Rotated Image", cmap='gray')
                self.__show_image(self.__resized_image, "Resized Image", cmap='gray')
                self.__show_image(self.__norm_face, "Cropped Image", cmap='gray')

            return self.__original_image, self.__norm_face
        
        except:
            return self.__original_image, None
        


if __name__ == "__main__":

    file_path = os.path.dirname(os.path.abspath(__file__))

    project_path = os.path.join(file_path, "..")
    original_path = os.path.join(project_path, "database\\originals\\")
    norm_path = os.path.join(project_path, "database\\normalized\\")

    for path, folders, files in os.walk(original_path):
        print("Inicialize normalization process...")

        if files == []:

            norm_path = os.path.join(norm_path, folders[0])
            if not os.path.exists(norm_path):
                print("Creating folder: ", norm_path)
                os.makedirs(norm_path)
                print("Created folder: ", norm_path)

        else: 
            for file in files: 

                print("Normalizing", file, "...")

                if(file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png")): 
                    image_path = os.path.join(path, file)
                    print("Image Path: ", image_path)
                    norm = Normalization(image_path)
                    original_image, normalized_image = norm.normalize(show_images=False)

                    norm.save_image(normalized_image, norm_path + file)
                    print("\n\n")
                else: 
                    print("File not supported: ", file)

        print("Normalization process finished.")
        
