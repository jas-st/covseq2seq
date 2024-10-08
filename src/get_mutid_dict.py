import pickle
import pandas as pd

path = "C:\\Users\\zhasmi00\\PycharmProjects\\mutation_database\\mutations_summary.txt"
df_ids = pd.read_csv(path, sep='\t', usecols=["Mutation ID", "Genome position", "Ref seq", "Mut seq"])

# create a new column that combines the values from 3 other columns in the A234G format
df_ids["nuc"] = df_ids["Ref seq"] + df_ids["Genome position"].astype(str) + df_ids["Mut seq"]
id_dict = dict(zip(df_ids["Mutation ID"], df_ids["nuc"]))

with open("id_dict.pkl", "wb") as file:
    pickle.dump(id_dict, file, pickle.HIGHEST_PROTOCOL)

nuc_dict = df_ids.groupby("Genome position").apply(lambda x: dict(zip(x["nuc"], x["Mutation ID"]))).to_dict()

with open("id_dict_reverse.pkl", "wb") as file:
    pickle.dump(nuc_dict, file, pickle.HIGHEST_PROTOCOL)
