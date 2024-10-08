import pandas as pd

path_res = "C:\\Users\\zhasmi00\\Downloads\\test\\august\\evols2022lineages.csv"
path_reduced = "C:\\Users\\zhasmi00\\Downloads\\test\\august\\evols2022lineages_reduced.xlsx"
path_trimmed = "C:\\Users\\zhasmi00\\Downloads\\test\\august\\evols2022lineages_cleaned.xlsx"
df_res = pd.read_csv(path_res, sep=';', usecols=["seqName", "partiallyAliased", "substitutions", "qc.overallStatus"])

ref_seq = ""
prev_idx = ""
reduce_mode = False

if reduce_mode:

    for idx in df_res.index:
        row = df_res.loc[idx]

        if row["qc.overallStatus"] != "good":
            df_res.drop(idx, inplace=True)
            continue

        if "input" in row["seqName"]:
            ref_seq = row["partiallyAliased"]
            continue

        if row["partiallyAliased"] == ref_seq:
            try:
                df_res.drop(idx-1, inplace=True)
            except KeyError:
                continue
            if "evol5" in row["seqName"]:
                df_res.drop(idx, inplace=True)

        ref_seq = row["partiallyAliased"]

    df_res.to_excel(path_reduced)


df_red = pd.read_excel(path_reduced)
# with expand the split elements will expand out into separate columns
df_red["evolID"] = df_red["seqName"].str.split("_", expand=True)[1]
df_red = df_red[df_red.duplicated("evolID", keep=False)]
df_red.to_excel(path_trimmed)