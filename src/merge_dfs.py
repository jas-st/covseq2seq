import pandas as pd

df_ref_path = "C:\\Users\\zhasmi00\\Downloads\\test\\RESULTS\\AugustKW29_22_beam\\evolutions_ausgewertet_2.csv"
df_cleaned_path = "C:\\Users\\zhasmi00\\Downloads\\test\\RESULTS\\AugustKW29_22_beam\\gisaid_removed_2.csv"

df_ref = pd.read_csv(df_ref_path, sep=';', index_col="seqName",
                     usecols=["seqName", "partiallyAliased", "qc.overallStatus",
                              "qc.privateMutations.status", "substitutions",
                              "privateNucMutations.unlabeledSubstitutions"])

df_cleaned = pd.read_csv(df_cleaned_path, sep=';', index_col="seqName",
                         usecols=["seqName", "partiallyAliased", "qc.overallStatus",
                                  "qc.privateMutations.status", "substitutions",
                                  "privateNucMutations.unlabeledSubstitutions"])

df_cleaned.drop(df_cleaned.loc[df_cleaned['qc.overallStatus'] != "good"].index, inplace=True)
df_ref2 = df_ref.drop(df_ref.loc[df_ref['qc.overallStatus'] != "good"].index)

merged_df = pd.concat([df_ref2, df_cleaned])
merged_df = merged_df[~merged_df.index.str.contains("input")]

merged_df.to_csv("C:\\Users\\zhasmi00\\Downloads\\test\\RESULTS\\AugustKW29_22_beam\\merged_iter2.csv", sep=";")

df_ref["evolID"] = df_ref.index.to_series().str.split("_", expand=True)[1]
df_cleaned["evolID"] = df_cleaned.index.to_series().str.split("_", expand=True)[1]

df_ref = df_ref.drop_duplicates(["partiallyAliased", "evolID"])
df_cleaned = df_cleaned.drop_duplicates(["partiallyAliased", "evolID"])

df_ref = df_ref.drop(df_ref[df_ref['qc.overallStatus'] != 'good'].index)
merged_df_for_validation = pd.concat([df_ref, df_cleaned])
merged_df_for_validation = merged_df_for_validation[~merged_df_for_validation.index.str.contains("input")]

merged_df_for_validation.to_csv("C:\\Users\\zhasmi00\\Downloads\\test\\RESULTS\\AugustKW29_22_beam\\merged_val_iter2"
                                ".csv", sep=";")
