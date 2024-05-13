import os
import cv2


class FirstDatabase:

    ORIGINALS = "first_assignment\\first_database\\originals\\"
    NORMALIZED = "first_assignment\\first_database\\normalized\\"
    EXAMPLES = "first_assignment\\first_database\\examples\\"
    RESOURCES = "first_assignment\\first_database\\resources\\"

    @staticmethod
    def store_normalized(image, title):
        project_source = FirstDatabase.NORMALIZED
        cv2.imwrite(project_source + title, image)

    @staticmethod
    def load(mode):
        project_source = None
        if mode == 'examples':
            project_source = FirstDatabase.EXAMPLES
        elif mode == 'originals':
            project_source = FirstDatabase.ORIGINALS
        elif mode == 'normalized':
            project_source = FirstDatabase.NORMALIZED
        elif mode == 'resources':
            project_source = FirstDatabase.RESOURCES
        absolute_path = "{}\\{}".format(os.getcwd(), project_source)
        images = []
        for name in os.listdir(absolute_path):
            image_path = project_source + name
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            images.append((image, name))
            print("Loaded {} image: {}.".format(mode, name))
        return images