import pickle
import pandas as pd

path = "C:\\Users\\zhasmi00\\PycharmProjects\\mutation_database\\correlations.txt"
path2 = "C:\\Users\\zhasmi00\\PycharmProjects\\mutation_database\\mutations_summary.txt"


with open("data/data_lang_full.pkl", "rb") as fo:
    data_lang = pickle.load(fo)

df = pd.read_csv(path, sep="\t", usecols=["Mutation A", "Mutation B", "Correlation_coefficient"])
df_ids = pd.read_csv(path2, sep='\t', usecols=["Mutation ID", "Genome position", "Ref seq", "Mut seq",
                                               "Protein mutation-1 letter"])

df_ids["nuc"] = df_ids["Ref seq"] + df_ids["Genome position"].astype(str) + df_ids["Mut seq"]
id_dict = dict(zip(df_ids["Mutation ID"], df_ids["nuc"]))
prot_dict = dict(zip(df_ids["Protein mutation-1 letter"], df_ids["Mutation ID"]))

# with open("id_nuc_dict.pkl", "wb") as file:
#     pickle.dump(id_dict, file, pickle.HIGHEST_PROTOCOL)
#
# with open("prot_id_dict.pkl", "wb") as file:
#     pickle.dump(prot_dict, file, pickle.HIGHEST_PROTOCOL)

# df["Mutation A"] = df["Mutation A"].apply(lambda x: (id_dict[x]))
# df["Mutation B"] = df["Mutation B"].apply(lambda x: (id_dict[x]))
#
# df["Mutation B"] = df[["Mutation B", "Correlation_coefficient"]].agg(tuple, axis=1)
# df.drop(["Correlation_coefficient"], inplace=True, axis=1)
# df = df.groupby("Mutation A").agg(list)
#
# corr_dict = dict(zip(df.index, df["Mutation B"]))
# corr_dict2 = {}
#
# for key, value in corr_dict.items():
#     corr_dict2[key] = dict(value)
# print(corr_dict2)

# with open("corr_dict.pkl", "wb") as file:
#     pickle.dump(corr_dict2, file, pickle.HIGHEST_PROTOCOL)



