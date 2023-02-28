import numpy as np
import skimage


class MorphologicalFeatureExtractor(object):
    def __init__(self):
        pass

    def extract(self, image):
        gray_image = skimage.color.rgb2gray(image)

        # Extract morphological features using the gray image
        selem = skimage.morphology.disk(5)  # Define a disk-shaped structuring element for morphological operations
        erosion = skimage.morphology.erosion(gray_image, selem)
        dilation = skimage.morphology.dilation(gray_image, selem)
        # opening = skimage.morphology.opening(gray_image, selem)
        # closing = skimage.morphology.closing(gray_image, selem)
        morph_features = np.stack((erosion, dilation), axis=0)
        # morph_features = np.stack((erosion, dilation, opening, closing), axis=0)
        return morph_features
