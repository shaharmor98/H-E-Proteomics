import os.path

from logger import Logger


class TilesLabeler(object):
    def __init__(self, rnr_to_metadata):
        self._ids_to_pam50 = rnr_to_metadata.map_id_to_pam50()
        self._logger = Logger()

    def attach_label_to_tile(self, tile_name):
        # assuming each tile is saved in the following format:
        # <name>.ndpi_x_y.jpeg
        rnr_id = tile_name[:tile_name.find(".")]

        if rnr_id not in self._ids_to_pam50:
            self._logger.info("Unknown rnr id: " + str(rnr_id))
            self._logger.info("Tile name: " + str(tile_name))
            raise RuntimeError()

        return self._ids_to_pam50[rnr_id]
