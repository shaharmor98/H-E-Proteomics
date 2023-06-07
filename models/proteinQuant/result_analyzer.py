import numpy as np

from data_parser.dia_to_metadata_parser import DiaToMetadata
from host_configuration import HostConfiguration
from torchvision import transforms
import json

from models.proteinQuant.tiles_dataset import TilesDataset


class ResultsAnalyzer(object):
    def __init__(self, gene_name):
        self.gene_name = gene_name
        self.tiles_directory = HostConfiguration.TILES_DIRECTORY.format(zoom_level=HostConfiguration.ZOOM_LEVEL,
                                                                        patch_size=HostConfiguration.PATCH_SIZE)
        self.transform_compose = transforms.Compose([transforms.Resize(size=(299, 299)),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize(mean=[0.], std=[255.])])
        self.dia_metadata = DiaToMetadata(HostConfiguration.DIA_GENES_FILE_PATH,
                                          HostConfiguration.RNR_METADATA_FILE_PATH,
                                          self.tiles_directory)
        self.gene_normalized_records = self.dia_metadata.get_normalized_gene_records(gene_name=gene_name)
        self.test_ids = self._get_test_ids()

    def predict_test_slides_expression(self):
        with open(HostConfiguration.PREDICTIONS_SUMMARY_FILE.format(gene=self.gene_name), 'r') as f:
            predictions = json.load(f)

        preds = []
        for test_id in self.test_ids:
            pred = predictions[test_id[0]]
            dataset = TilesDataset(self.tiles_directory, self.transform_compose, [test_id], caller="Prediction dataset")
            threshold_predictions = np.where(np.asarray(pred) > 0.5, 1, 0)
            averaged_tile = np.mean(threshold_predictions, axis=0)
            total = np.sum(np.where(np.asarray(averaged_tile) > 0.5, 1, 0), axis=0)
            ratio = total / dataset.get_num_of_files()
            preds.append(ratio)

        return np.asarray(preds)

    def get_actual_test_slides_expression(self):
        actual_prediction = []
        for slide_id, _ in self.test_ids:
            actual_prediction.append(self.gene_normalized_records[slide_id])

        return np.asarray(actual_prediction)

    def _get_test_ids(self):
        with open(HostConfiguration.TEST_IDS_FILE.format(gene=self.gene_name), 'r') as f:
            ids = json.load(f)
            test_ids = []
            for k, v in ids.items():
                test_ids.append((k, v))

        return test_ids


"""
Open issues:
1. How should I get predicted expression value? suppose I have 9 tiles (3 by 3), each has a score between zero to one
Then I round greater than 0.5 to 1 and less than 0.5 to 0. Then what ? Zohar proposed to sum the 1's and divide by 9.
Other suggestions ? 
2. About ROC- what do I need to do ? E.g, I take one test id actual expression (float) and round up to be 1 or 0 ?
Or leave it as it is ? if I round then I can use ROC, otherwise how can I do it ?
3. Confusion matrix - I have float values, should I round them up ? should I use the middle value as the threshold ? 
4. Ilan had the same task ? If so should I just adjust my code to his ?
5. Ilan- some explanation about input formats in the evaluation utils code ?  
"""