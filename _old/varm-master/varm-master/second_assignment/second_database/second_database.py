
import os
import re
import cv2


class SecondDatabase:

    EXAMPLES = "second_assignment\\second_database\\examples\\"
    RESULTS = "second_assignment\\second_database\\results\\"
    OBJECTS = "second_assignment\\second_database\\objects\\"

    @staticmethod
    def load(mode, listing=False):
        project_source = None
        if mode == 'examples':
            project_source = SecondDatabase.EXAMPLES
        elif mode == 'objects':
            project_source = SecondDatabase.OBJECTS
        absolute_path = "{}\\{}".format(os.getcwd(), project_source)
        images = {}
        for count, name in enumerate(os.listdir(absolute_path)):
            image_path = project_source + name
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            index = int(re.search(r'\d+', name).group()) if listing else count
            images[index] = image
            print("Loaded {} image: {}.".format(mode, name))
        return images

