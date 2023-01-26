import os
import random

import numpy as np
import pandas as pd

from host_configuration import HostConfiguration

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
        full_genes = self._dia_df.loc[condition, ["Gene_symbol"] + target_column_names]
        full_genes = full_genes.reset_index()
        return full_genes

    def get_gene_normalized_protein_quant(self, gene_row):
        cols = [col for col in gene_row.columns if col.startswith("ProteinQuant_")]

        # normalized_cols = ["MinMaxNorm" + col for col in self._dia_df.columns if col.startswith("ProteinQuant_")]
        # row = self._dia_df.loc[self._dia_df.index == gene_index, cols]

        def normalize(row):
            return (row - row.min()) / (row.max() - row.min())

        gene_row = gene_row[cols].apply(normalize, axis=1)
        return gene_row

    def get_upper_lower_percentage(self, row):
        high_percentile = np.percentile(row, 80)
        low_percentile = np.percentile(row, 20)

        high_cols = row[row >= high_percentile].dropna(axis=1).columns
        low_cols = row[row <= low_percentile].dropna(axis=1).columns

        return high_cols, low_cols, high_percentile, low_percentile

    def get_tnbc_unique_df(self):
        tnbc = self._metadata_df[self._metadata_df['TNBC'] == 1]  # Focusing only on positive TNBC cases
        duplicated_ids = tnbc.duplicated(subset=['SCANB_PD_ID'])  # Assuming SCANB_PD_ID is unique- otherwise, fail.
        if duplicated_ids.isin([True]).any():
            print("Found duplicated SCANB_PD_ID values")
            raise RuntimeError("Duplicated columns found, abort")

        return tnbc

    def get_existing_slides_ids(self):
        potential_ids = {}
        for img in os.listdir(self.tiles_directory):
            potential_ids[img[:img.find(".")]] = 1

        return potential_ids

    def get_gene_slides_with_labels(self, gene_name):
        tnbc = self.get_tnbc_unique_df()
        ids = self.get_existing_slides_ids()
        tnbc = tnbc[tnbc['SCANB_PD_ID'].isin(ids.keys())]
        rnrs = self.map_scanb_pd_id_to_rnr(tnbc['SCANB_PD_ID'].to_list())
        genes = self.get_genes_with_complete_records(rnrs)
        gene_row = genes.loc[genes['Gene_symbol'] == gene_name]
        if len(gene_row) != 1:
            raise RuntimeError("WTF just happened with {}".format(gene_name))

        normalized_row = self.get_gene_normalized_protein_quant(gene_row)
        high_cols, low_cols, _, _ = self.get_upper_lower_percentage(normalized_row)

        target_rnrs = [(record[record.find("_") + 1:], 1) for record in high_cols.to_list() if
                       record.startswith('ProteinQuant_')]
        target_rnrs += [(record[record.find("_") + 1:], 0) for record in low_cols.to_list() if
                        record.startswith('ProteinQuant_')]

        slide_ids = {}
        for rnr, label in target_rnrs:
            slide_id = list(filter(lambda x: rnrs[x] == rnr, rnrs))[0]
            slide_ids[slide_id] = label

        return slide_ids

    def analyze_genes(self):
        print("Starting with: {} rows".format(len(self._metadata_df)))
        tnbc = self.get_tnbc_unique_df()
        print("Continuing with {} tnbc slides".format(len(tnbc)))
        ids = self.get_existing_slides_ids()
        print("Found {} existing slides".format(len(ids)))
        tnbc = tnbc[tnbc['SCANB_PD_ID'].isin(ids.keys())]
        print("Continuing with {} existing slides".format(len(tnbc)))
        rnrs = self.map_scanb_pd_id_to_rnr(tnbc['SCANB_PD_ID'].to_list())
        genes = self.get_genes_with_complete_records(rnrs)
        print("Found {} complete genes, start analysing".format(len(genes)))

        results = {}
        summary_results = {}

        for row_number in range(len(genes)):
            gene_row = genes.loc[genes.index == row_number]
            gene_name = gene_row['Gene_symbol'].values[0]
            print("Gene: {}".format(gene_name))
            normalized_row = self.get_gene_normalized_protein_quant(gene_row)
            high_cols, low_cols, high_percentile, low_percentile = self.get_upper_lower_percentage(normalized_row)
            results[gene_name] = [high_cols, low_cols]
            summary_results[gene_name] = [high_percentile, low_percentile]

        # try to sort by those whose 20% and 80% are the most significant..

        """
        dist_summary = {}                                              
        for k,v in summary_results.items():                                                       
            high, low = v[0], v[1]                                                    
            distance = high-low                                                        
            dist_summary[k] = [distance, high, low]
        sorted_dict = dict(sorted(dist_summary.items(), key=lambda x: x[1][0], reverse=True))
        """
        """
        Gene: CC2D1A, distance: 0.45659020244730475; high: 0.5464200181355323; low: 0.08982981568822758
        Gene: IL1RL1, distance: 0.4313145627292086; high: 0.5538109614983974; low: 0.12249639876918879
        Gene: FLNA, distance: 0.4233408823076643; high: 0.5841377008192563; low: 0.160796818511592
        Gene: ANP32B, distance: 0.41526041861039276; high: 0.5265247184527895; low: 0.1112642998423967
        Gene: TPM3, distance: 0.4117285671735882; high: 0.4922182676882266; low: 0.08048970051463837
        Gene: PLEKHO2, distance: 0.41041719006617156; high: 0.6080892474882015; low: 0.19767205742202992
        Gene: CCDC134, distance: 0.40758713121307827; high: 0.5583495832542409; low: 0.15076245204116262
        Gene: RPS15A, distance: 0.4069616364820037; high: 0.6672253662265061; low: 0.2602637297445024
        Gene: COX5A, distance: 0.4066843127938558; high: 0.6776681937126363; low: 0.2709838809187805
        Gene: PABPN1, distance: 0.4060252181494216; high: 0.5788087136823336; low: 0.17278349553291192
        Gene: STAT1, distance: 0.40545945263553945; high: 0.5408551981801274; low: 0.13539574554458794
        Gene: CFL1, distance: 0.4032360800565875; high: 0.7447725295044771; low: 0.34153644944788963
        Gene: EIF3A, distance: 0.40245241403758714; high: 0.5020887156514047; low: 0.09963630161381751
        Gene: PSMD8, distance: 0.4023671938715052; high: 0.6321002455952025; low: 0.2297330517236973
        Gene: EIF2S3, distance: 0.39926495017183794; high: 0.6527281073845488; low: 0.25346315721271084
        Gene: MBNL1, distance: 0.3982878159242543; high: 0.5942889709570055; low: 0.19600115503275123
        Gene: PACS1, distance: 0.3942722554301032; high: 0.4990865782763056; low: 0.1048143228462024
        Gene: HLA-C, distance: 0.38670822471874; high: 0.48883242492037154; low: 0.10212420020163153
        Gene: NFKB2, distance: 0.38429598669478443; high: 0.4768618062047525; low: 0.09256581950996805
        Gene: RPS9, distance: 0.3820729407982868; high: 0.6142004822946795; low: 0.2321275414963927
        """
        return results, summary_results

    def random_shuffle(self, gene_slides_with_labels, test_proportion=0.1):
        high_instances = [slide for slide, label in gene_slides_with_labels.items() if label == 1]
        low_instances = [slide for slide, label in gene_slides_with_labels.items() if label == 0]

        test_size = int(len(high_instances) * test_proportion)

        random.seed(HostConfiguration.SEED)
        random.shuffle(high_instances)
        random.shuffle(low_instances)

        train_ids = {}
        test_ids = {}

        for instance in high_instances[:test_size]:
            test_ids[instance] = 1
        for instance in high_instances[test_size:]:
            train_ids[instance] = 1
        for instance in low_instances[:test_size]:
            test_ids[instance] = 0
        for instance in low_instances[test_size:]:
            test_ids[instance] = 0

        return train_ids, test_ids
