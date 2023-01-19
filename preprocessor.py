import glob
import os
import pathlib
from multiprocessing import Pool

from data.tiles.tiles_extractor import TilesExtractor
from host_configuration import HostConfiguration


class Preprocessor(object):
    SLIDES_FORMAT_NAME = ".ndpi"

    def __init__(self, slides_directory, zoom=HostConfiguration.ZOOM_LEVEL, patch_size=HostConfiguration.PATCH_SIZE):
        self._slides_directory = slides_directory
        self._zoom = zoom
        self._patch_size = patch_size
        self._setup()
        tiles_directory = HostConfiguration.TILES_DIRECTORY.format(zoom_level=self._zoom, patch_size=self._patch_size)
        self._tiles_extractor = TilesExtractor(zoom=zoom, patch_size=patch_size,
                                               tiles_directory=tiles_directory)

    def start(self):
        slides_path = self._get_slides_path()

        with Pool(HostConfiguration.CPU_CORES_NUM) as p:
            p.map(self._tiles_extractor.extract, slides_path)

    def _setup(self):
        images_directory = HostConfiguration.TILES_DIRECTORY.format(zoom_level=self._zoom, patch_size=self._patch_size)
        if not os.path.exists(images_directory):
            os.makedirs(images_directory)

    def _get_slides_path(self):
        paths = []
        print("Iterating slides directory: ", self._slides_directory)

        for file_path in glob.glob(os.path.join(self._slides_directory, "*")):
            file_suffix = os.path.splitext(file_path)[1]
            if file_suffix == "" or file_suffix != self.SLIDES_FORMAT_NAME:
                continue

            paths.append(file_path)

        return paths
