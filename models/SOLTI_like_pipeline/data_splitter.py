import random

import numpy as np


class DataSplitter(object):
    def __init__(self, dia):
        self.dia = dia

    def split_train_val(self, instances, seed=42, split_size=0.3):
        """
        Expecting input to be a list of lists, first element is slide name, second is slide label
        """

        high_instances = [slide for slide, label in instances.items() if label == 1]
        low_instances = [slide for slide, label in instances.items() if label == 0]

        validation_set_size = int(len(high_instances) * split_size)

        random.seed(seed)
        random.shuffle(high_instances)
        random.shuffle(low_instances)

        train_records = []
        validation_records = []

        for instance in high_instances[:validation_set_size]:
            validation_records.append((instance, 1))
        for instance in high_instances[validation_set_size:]:
            train_records.append((instance, 1))
        for instance in low_instances[:validation_set_size]:
            validation_records.append((instance, 0))
        for instance in low_instances[validation_set_size:]:
            train_records.append((instance, 0))

        return train_records, validation_records
