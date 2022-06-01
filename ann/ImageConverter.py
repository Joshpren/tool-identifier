import os
import cv2

class ImageConverter:

    IMG_SIZE = 100

    def convert(self, directory, categories):
        data = []
        for category in categories:
            path = os.path.join(directory, category)
            class_num = categories.index(category)
            for img in os.listdir(path):
                try:
                    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                    new_array = cv2.resize(img_array, (self.IMG_SIZE, self.IMG_SIZE))
                    data.append([new_array, class_num])
                except Exception as e:
                    pass
        return data
