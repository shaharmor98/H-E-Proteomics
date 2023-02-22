import os


class HostConfiguration(object):
    CPU_CORES_NUM = 60
    SLIDES_DIRECTORY = "../data/slides"
    TILES_DIRECTORY = os.path.join(os.path.expanduser("~"), "proteomics-tiles-shahar-mor",
                                   "images/zoom_{zoom_level}_size_{patch_size}")
    RNR_METADATA_FILE_PATH = os.path.join(os.path.expanduser("~"), "proteomics_from_janne", "Labels_etc",
                                          "230109_RNr_to_metadata.xlsx")
    DIA_GENES_FILE_PATH = os.path.join(os.path.expanduser("~"), "proteomics_from_janne", "Labels_etc",
                                       "SCANB_TNBC_diaPASEF_SN16_E104.xlsx")
    PATCH_SIZE = 512
    ZOOM_LEVEL = 20
    SEED = 42
    CHOSEN_GENE = "STAT1"
    # CHOSEN_GENE = "HLA-C"
    # CHOSEN_GENE = "NFKB2"
    GENES = ["STAT1", "CC2D1A", "IL1RL1", "MAPK13", "ATP5MK", "HDAC1"]
    CHECKPOINTS_PATH = os.path.join(os.path.expanduser("~"), "checkpoints", "{gene}")
    PREDICTIONS_SUMMARY_FILE = os.path.join(CHECKPOINTS_PATH, "predictions.txt")
    TEST_IDS_FILE = os.path.join(CHECKPOINTS_PATH, "test_ids.txt")
    NUM_OF_FOLDS = 10
