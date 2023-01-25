import os

import pandas as pd

"""
You should implement the following things:
5. Train on that gene- should be 0 or 1. filter out records as Alona did
6. Create test function that unites everything 
"""


class DiaToMetadata(object):
    def __init__(self, dia_excel_path, metadata_excel_path, tiles_directory):
        self._dia_df = pd.read_excel(dia_excel_path, usecols=range(300))
        self._metadata_df = pd.read_excel(metadata_excel_path)
        self.tiles_directory = tiles_directory

    def map_scanb_pd_id_to_rnr(self, pd_ids):
        rnrs = {}

        for pd_id in pd_ids:
            res = self._metadata_df[self._metadata_df['SCANB_PD_ID'] == pd_id]
            if len(res) != 1:
                raise RuntimeError("Can't parse pd_ids to rnr, failed for: {} id".format(pd_id))

            rnrs[pd_id] = res['RNr'].values[0]

        return rnrs

    def get_genes_with_complete_records(self, rnrs):
        target_column_names = []
        for rnr in rnrs.values():
            target_column_names.append("ProteinQuant_" + rnr)

        condition = self._dia_df[target_column_names].notnull().all(axis=1)
        full_genes = self._dia_df.loc[condition, [["Gene_symbol"] + target_column_names]]
        return full_genes

    def get_gene_normalized_protein_quant(self, gene_index):
        cols = [col for col in self._dia_df.columns if col.startswith("ProteinQuant_")]
        # normalized_cols = ["MinMaxNorm" + col for col in self._dia_df.columns if col.startswith("ProteinQuant_")]
        row = self._dia_df.loc[self._dia_df.index == gene_index, cols]

        def normalize(row):
            return (row - row.min()) / (row.max() - row.min())

        row = row[cols].apply(normalize, axis=1)
        return row

    def print_upper_lower_percentage(self, row):
        high_threshold = 0.8
        low_threshold = 0.2

        high_cols = row.columns[row.gt(high_threshold).any()]
        low_cols = row.columns[row.lt(low_threshold).any()]

        print("Found {} samples with high gene amount".format(len(high_cols)))
        print("Found {} samples with low gene amount".format(len(low_cols)))

        return high_cols, low_cols

    def get_tnbc_unique_df(self):
        tnbc = self._metadata_df[self._metadata_df['TNBC'] == 1]  # Focusing only on positive TNBC cases
        duplicated_ids = tnbc.duplicated(subset=['SCANB_PD_ID'])  # Assuming SCANB_PD_ID is unique- otherwise, fail.
        if duplicated_ids.isin([True]).any():
            print("Found duplicated SCANB_PD_ID values")
            raise RuntimeError("Duplicated columns found, abort")

        return tnbc

    def get_existing_slides_ids(self):
        potential_ids = set()
        for img in os.listdir(self.tiles_directory):
            potential_ids.add(img[:img.find(".")])

        return potential_ids
