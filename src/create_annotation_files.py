import pandas as pd
import utils


def trim_lin(lin):

    if lin in top_lins:
        return lin+".*"
    elif len(lin.split(".")) == 1:
        return lin+".*"
    else:
        lin_new = lin.split(".")[:-1]

        while (".".join(lin_new) not in top_lins) and (len(lin_new) != 1):
            lin_new = lin_new[:-1]

        return ".".join(lin_new)+".*"


path_df = "C:\\Users\\zhasmi00\\Downloads\\test\\august\\evols2022lineages_cleaned_1.xlsx"
df = pd.read_excel(path_df)

save_path_1 = "C:\\Users\\zhasmi00\\Downloads\\test\\august\\stripe_annotation.txt"
save_path_2 = "C:\\Users\\zhasmi00\\Downloads\\test\\august\\lines_annotation.txt"

group = df.groupby("partiallyAliased")["evolID"]
counts = group.count()
lin_prev = counts.divide(counts.sum())

# .str is used to access the values of the series as strings and apply methods to them
# .shift moves the column by minus one, so that cat can pair each instance with the next one
evol_pairs = df.groupby('evolID')["seqName"].apply(lambda x: x.str.cat(x.shift(-1), sep=',')).dropna()

top_lins = lin_prev[lin_prev.gt(0.01)].index

print(lin_prev.index.to_series().apply(lambda x: trim_lin(x)))







# with open(save_path_1, "w") as stripes:
#     for idx in df.index:
#         row = df.loc[idx]
#
#
#         stripes_string = row["seqName"] + ' ' + hex_codes[color] + ' ' + row["partiallyAliased"]
#
#         annot.write(annot_string + "\n")





