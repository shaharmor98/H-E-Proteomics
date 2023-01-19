import os


class HostConfiguration(object):
    CPU_CORES_NUM = 60
    SLIDES_DIRECTORY = "../data/slides"
    TILES_DIRECTORY = "../images/zoom_{zoom_level}_size_{patch_size}"
    RNR_METADATA_FILE_PATH = os.path.join(os.path.expanduser("~"), "Labels_etc_230109_RNr_to_metadata.xlsx")
    PATCH_SIZE = 512
    ZOOM_LEVEL = 20
