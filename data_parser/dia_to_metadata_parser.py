import pandas as pd


class DiaToMetadata(object):
    def __init__(self, excel_path):
        self._df = pd.read_excel(excel_path)

    def get_tnbc_unique_df(self):
        tnbc = self._df[self._df['TNBC'] == 1]  # Focusing only on positive TNBC cases
        duplicated_ids = tnbc.duplicated(subset=['SCANB_PD_ID'])  # Assuming SCANB_PD_ID is unique- otherwise, fail.
        if duplicated_ids.isin([True]).any():
            print("Found duplicated SCANB_PD_ID values")
            raise RuntimeError("Duplicated columns found, abort")

        # PAM50 subtype slides aren't equally distributed, so we will stick with the 3 top subtypes
        return tnbc

