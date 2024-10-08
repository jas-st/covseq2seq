import pandas as pd
import pickle
import os
import itertools

MONTHS = ["Jan", "Feb", "March", "April", "May", "June", "July", "August", "Sep", "Oct", "Nov", "Dec"]
YEARS = ["2020", "2021", "2022"]

ref_month = "August2022"
ref_week = "KW29"
path_data = "C:\\Users\\zhasmi00\\PycharmProjects\\DATA\\"


for ele in itertools.product(YEARS, MONTHS):
    path = path_data + ele[0] + "\\" + "".join([ele[1], ele[0]])
    for file in os.listdir(path):
        path2 = path + "\\" + file
        lins = [x[:-4] for x in os.listdir(path2) if "_summary" not in x]




quit()



for file in os.listdir(path):
    if file.endswith('.csv'):
        kw_dfs.append(file)



ref_df = pd.read_csv(ref1_path, sep=';', index_col="seqName", usecols=["seqName", "partiallyAliased", "substitutions"])
result_df = pd.read_csv(result1_path, sep=';', index_col="seqName", usecols=["seqName", "partiallyAliased", "substitutions"])

# every observed lineage
with open('lineage_set.pkl', "rb") as mut:
    lineage_set = pickle.load(mut)

ref_lineages = set(ref_df["partiallyAliased"].values)
result_lineages = set(result_df["partiallyAliased"].values)

print("MISSED")
print([x for x in ref_lineages if x not in result_lineages])
print("TREFFER")
print([x for x in result_lineages if x in ref_lineages])
print("NOVEL")
print([x for x in result_lineages if x not in lineage_set])

