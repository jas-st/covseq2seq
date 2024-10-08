import pandas as pd

file_path = "C:\\Users\\zhasmi00\\Downloads\\test\\RESULTS\\AugustKW29_22_beam\\evolutions_ausgewertet_4.csv"

df = pd.read_csv(file_path, sep=';', index_col="seqName", usecols=["seqName", "partiallyAliased", "qc.overallStatus",
                                                                   "qc.privateMutations.status", "substitutions",
                                                                   "privateNucMutations.unlabeledSubstitutions"])

df["evolID"] = df.index.to_series().str.split("_", expand=True)[1]

df.drop_duplicates(["partiallyAliased", "evolID"], inplace=True)
df = df[~df.index.str.contains("input")]

df.to_csv("C:\\Users\\zhasmi00\\Downloads\\test\\RESULTS\\AugustKW29_22_beam\\evolutions_ausgewertet_4_cleaned.csv", sep=";")

