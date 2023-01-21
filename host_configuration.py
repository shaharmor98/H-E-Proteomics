import os


class HostConfiguration(object):
    CPU_CORES_NUM = 60
    SLIDES_DIRECTORY = "../data/slides"
    TILES_DIRECTORY = os.path.join(os.path.expanduser("~"), "images/zoom_{zoom_level}_size_{patch_size}")
    RNR_METADATA_FILE_PATH = os.path.join(os.path.expanduser("~"),
                                          "proteomics_from_janne", "Labels_etc",
                                          "Labels_etc_230109_RNr_to_metadata.xlsx")
    PATCH_SIZE = 512
    ZOOM_LEVEL = 20
    SEED = 42
    CHECKPOINTS_PATH = os.path.join(os.path.expanduser("~"), "checkpoints")
