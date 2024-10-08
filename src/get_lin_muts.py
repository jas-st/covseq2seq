import pandas as pd
import pickle
import utils
from collections import defaultdict, Counter

path = "C:\\Users\\zhasmi00\\Downloads"

data_list = utils.get_data(path, 31)
ref_dict = None
desc_dict = defaultdict(list)

train_pairs = []
lineage_set = set()
lineage_mutations_dict = defaultdict(Counter)

for weekly_df in range(1, len(data_list)):
    print("Processing: ", data_list[weekly_df])
    print("Reference: ", data_list[weekly_df-1])

    if ref_dict is None:
        df_ref_path = data_list[weekly_df-1]
        df_ref_year = df_ref_path.split("_")[-1].split(".")[0][-4:]
        df_ref_month = df_ref_path.split("_")[-1].split(".")[0]

        df_ref = pd.read_csv(path + "\\" + df_ref_year + "\\" + df_ref_month + "\\" + df_ref_path,
                             sep=';', index_col="seqName",
                             usecols=["seqName", "partiallyAliased", "qc.overallStatus",
                                      "qc.privateMutations.status", "substitutions",
                                      "privateNucMutations.unlabeledSubstitutions"])

        # remove duplicates
        df_ref = df_ref[~df_ref.index.duplicated(keep='first')]

        # create ref dict
        ref_dict = utils.create_dictionary(df_ref)

    else:
        ref_dict = {k: v for k, v in desc_dict.items()}

    df_desc_path = data_list[weekly_df]
    df_desc_year = df_desc_path.split("_")[-1].split(".")[0][-4:]
    df_desc_month = df_desc_path.split("_")[-1].split(".")[0]

    df_desc = pd.read_csv(path + "\\" + df_desc_year + "\\" + df_desc_month + "\\" + df_desc_path,
                          sep=';', index_col="seqName",
                          usecols=["seqName", "partiallyAliased", "qc.overallStatus",
                                   "qc.privateMutations.status", "substitutions",
                                   "privateNucMutations.unlabeledSubstitutions"])

    # remove duplicates
    df_desc = df_desc[~df_desc.index.duplicated(keep='first')]

    # create desc dict
    desc_dict = utils.create_dictionary(df_desc)
