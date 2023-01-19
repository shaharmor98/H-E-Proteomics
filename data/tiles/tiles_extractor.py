import os

import numpy as np
import openslide
from PIL import Image


class TilesExtractor(object):
    def __init__(self, zoom, patch_size, tiles_directory):
        self._zoom = zoom
        self._patch_size = patch_size
        self._tiles_directory = tiles_directory

    def extract(self, slide_path):
        slide_name = os.path.basename(slide_path)

        try:
            slide = openslide.OpenSlide(slide_path)
            if openslide.PROPERTY_NAME_MPP_X in slide.properties:
                mag = 10.0 / float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
            elif "XResolution" in slide.properties:
                mag = 10.0 / float(slide.properties["XResolution"])
            elif 'tiff.XResolution' in slide.properties:  # for Multiplex IHC WSIs, .tiff images
                mag = 10.0 / float(slide.properties["tiff.XResolution"])
            else:
                print('[WARNING] mpp value not found. Assuming it is 40X with mpp=0.254!', slide_path)
                mag = 10.0 / float(0.254)
            pw = int(self._patch_size * mag / self._zoom)
            width = slide.dimensions[0]
            height = slide.dimensions[1]
        except Exception as e:
            print('{}: exception caught, message: {}'.format(slide_path, str(e)))
            return

        print(slide_path, width, height)
        for x in range(1, width, pw):
            for y in range(1, height, pw):
                if x + pw > width:
                    pw_x = width - x
                else:
                    pw_x = pw
                if y + pw > height:
                    pw_y = height - y
                else:
                    pw_y = pw

                if (int(self._patch_size * pw_x / pw) <= 0) or \
                        (int(self._patch_size * pw_y / pw) <= 0) or \
                        (pw_x <= 0) or (pw_y <= 0):
                    continue

                patch = slide.read_region((x, y), 0, (pw_x, pw_y))
                patch = patch.resize((self._patch_size * pw_x // pw, self._patch_size * pw_y // pw), Image.ANTIALIAS)
                patch = patch.convert('RGB')
                is_blank, pct_blank = self.check_img_is_blank(patch)
                if not is_blank:
                    image_name = '{}/{}_{}_{}.jpeg'.format(self._tiles_directory, slide_name, x // pw, y // pw)
                    patch.save(image_name)

        print("Extracted: ", slide_name)

    @staticmethod
    def check_img_is_blank(img):
        im = np.array(img)
        pct_bkg = np.mean((im > 220) * 1)
        if pct_bkg >= 0.5:
            return True, pct_bkg
        return False, pct_bkg
