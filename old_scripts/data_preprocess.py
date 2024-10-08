# this preprocess goes through all the raw csv data and only
# takes the sequences with "good" score into consideration
# saved in the lin_data dataframe where each lineage and its mutation dict are a row

from seq2seq_full.utils import get_data
import pandas as pd

file_dir = "/home/scotty/Jas_Experiments/gis_data/gis_files/"
data_list = get_data(None, file_dir)
complete_training_data = pd.DataFrame()
lin_dict = {}


for weekly_df in data_list:
    print("Processing: ", weekly_df)
    week = weekly_df.split("_")[1]
    month = weekly_df.split("_")[-1].split(".")[0]

    kw_csv = pd.read_csv(file_dir + month + "/" + weekly_df, sep=';', index_col="seqName",
                         usecols=["seqName", "partiallyAliased", "qc.overallStatus", "substitutions"])

    # remove duplicates
    kw_csv = kw_csv[~kw_csv.index.duplicated(keep='first')]

    for name in kw_csv.index:
        if kw_csv.loc[name, 'qc.overallStatus'] != "good":
            continue
        row = kw_csv.loc[name]
        lin = row["partiallyAliased"]

        if type(row["substitutions"]) != float:
            subs = row["substitutions"].split(",")
        else:
            subs = []

        if lin in lin_dict:
            lin_dict[lin]['count'] += 1
            for sub in subs:
                try:
                    lin_dict[lin]['muts'][sub] += 1
                except KeyError:
                    lin_dict[lin]['muts'][sub] = 1
        else:
            lin_dict[lin] = {'muts': dict.fromkeys(subs, 1), 'count': 1}  # , 'month': month+'_'+week}

# save to a csv file
lin_df = pd.DataFrame.from_dict(lin_dict, orient='index').rename_axis('lineage').reset_index()
lin_df.set_index("lineage", inplace=True)
lin_df.to_csv("lin_data.csv")
