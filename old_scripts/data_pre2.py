from seq2seq_full.utils import get_data
import pandas as pd
from collections import defaultdict
import pickle

file_dir = "/home/scotty/Jas_Experiments/gis_data/gis_files/"
ref_df = pd.read_csv("lin_data.csv")
ref_df.set_index("lineage", inplace=True)

# relevant_muts = defaultdict()
#
# for lin in ref_df.index.values:
#     mut_dict = ast.literal_eval(ref_df.loc[lin, "muts"])
#     relevant_muts[lin] = [x for x in mut_dict.keys() if mut_dict[x]/ref_df.loc[lin, "count"] > 0.35]
#
# with open("relevant_muts.json", "wb") as fp:
#     pickle.dump(relevant_muts, fp, protocol=pickle.HIGHEST_PROTOCOL)

with open('relevant_muts.json', "rb") as mut:
    relevant_muts = pickle.load(mut)
data_list = get_data(None, file_dir)
complete_training_data = pd.DataFrame()
lin_dict = defaultdict(set)

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

        if lin[0] == "X" or lin[:2] == "BA" or lin in ["B.1", "B", "A", "B.1.1"]:
            continue

        if type(row["substitutions"]) != float:
            subs = row["substitutions"].split(",")
            subs = [mut for mut in subs if mut in relevant_muts[lin]]
            subs = frozenset(subs)
        else:
            subs = frozenset()

        lin_dict[lin].add(subs)
    # print(pd.DataFrame.from_dict(lin_dict, orient="index").rename_axis("lineage"))

# save to a df file
lin_df = pd.DataFrame.from_dict(lin_dict, orient="index").rename_axis("lineage")
lin_df.to_csv("lin_muts.csv")


