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
        self._tiles_extractor = TilesExtractor(zoom=zoom, patch_size=patch_size,
                                               tiles_directory=HostConfiguration.TILES_DIRECTORY)

    def start(self):
        slides_path = self._get_slides_path()

        print("Slides_path: ", slides_path)
        exit(1)
        
        with Pool(HostConfiguration.CPU_CORES_NUM) as p:
            p.map(self._tiles_extractor.extract, slides_path)

    def _setup(self):
        images_directory = HostConfiguration.TILES_DIRECTORY.format(zoom_level=self._zoom, patch_size=self._patch_size)
        if not os.path.exists(images_directory):
            os.makedirs(images_directory)

    def _get_slides_path(self):
        paths = []

        for file_name in os.listdir(self._slides_directory):
            file_path = pathlib.Path(file_name)
            if file_path.suffix == "" or file_path.suffix != self.SLIDES_FORMAT_NAME:
                continue

            paths.append(str(file_path.absolute()))

        return paths
