import numpy as np
import skimage
from skimage.color import gray2rgb
from skimage.feature import local_binary_pattern, graycoprops, graycomatrix


class TextureFeaturesExtractor(object):
    def __init__(self):
        pass

    def extract(self, image):
        gray_image = skimage.color.rgb2gray(image)

        # compute LBP features
        lbp_features = self.compute_lbp(gray_image, radius=3, n_points=8)

        # compute GLCM features
        glcm_features = self.compute_glcm(gray_image)

        # concatenate features into a single feature vector
        texture_features = np.concatenate([lbp_features, glcm_features])
        return texture_features

    # define LBP function
    def compute_lbp(self, gray_image, radius, n_points):
        lbp = local_binary_pattern(gray_image, n_points, radius, method='uniform')
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
        return hist

    # define GLCM function
    def compute_glcm(self, image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True):
        img_uint = (image * 255).astype(np.uint8)
        glcm = graycomatrix(img_uint, distances, angles, levels=levels, symmetric=symmetric, normed=normed)
        features = {'contrast': [], 'dissimilarity': [], 'homogeneity': [], 'energy': [], 'correlation': []}
        for prop in features.keys():
            features[prop] = [graycoprops(glcm, prop) for angle in angles for distance in
                              distances]
        res = np.concatenate(list(features.values()))
        res = res.reshape(-1, )

        return res
