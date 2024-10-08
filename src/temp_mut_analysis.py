import pandas as pd
from collections import defaultdict

path_res = "C:\\Users\\zhasmi00\\Downloads\\test\\august\\evols2022august.csv"
path_csv = "C:\\Users\\zhasmi00\\Downloads\\test\\august\\heatmap10.xlsx"
results_df = pd.read_csv(path_res, sep=';', usecols=["seqName", "substitutions",
                                                     "privateNucMutations.unlabeledSubstitutions"])

heatmap_mutations = defaultdict(lambda: defaultdict(int))
input_mutations = defaultdict(lambda: 0)
evolutioner = []

for idx in results_df.index:
    row = results_df.loc[idx]

    if row["seqName"] == "input":
        evolutioner = row["substitutions"].split(",")

        if type(row["privateNucMutations.unlabeledSubstitutions"]) != float:
            private_muts = row["privateNucMutations.unlabeledSubstitutions"].split(",")
            evolutioner = [x for x in evolutioner if x not in private_muts]
        continue

    try:
        evolutionee_raw = row["substitutions"].split(",")
    except AttributeError:
        continue

    if type(row["privateNucMutations.unlabeledSubstitutions"]) != float:
        private_muts = row["privateNucMutations.unlabeledSubstitutions"].split(",")
        evolutionee_raw = [x for x in evolutionee_raw if x not in private_muts]
    evolutionee = [x for x in evolutionee_raw if x not in evolutioner]

    for mut in evolutioner:
        input_mutations[mut] += 1
        for evol in evolutionee:
            heatmap_mutations[mut][evol] += 1

    if row["seqName"] == "evol5":
        evolutioner = []
    else:
        evolutioner = [x for x in evolutionee_raw]

input_mutations_df = pd.DataFrame(input_mutations.items(), columns=["mutation", "amount"])
input_mutations_df.set_index("mutation", inplace=True)


df = pd.DataFrame(heatmap_mutations)
df.sort_index(key=lambda x: x.str[1:-1].astype(int), inplace=True)
df.fillna(0, inplace=True)

full_count_ref_df = pd.DataFrame(heatmap_mutations)

for idx in input_mutations_df.index:
    amnt = input_mutations_df.loc[idx]["amount"]
    try:
        df[idx] = (df[idx] / amnt).round(2)
    except KeyError:
        continue

df = df.astype(float)
df = df.where(df >= 0.1).dropna(how='all').dropna(axis=1, how='all')
mut_tupels = []


for evolvee in df.index:
    row = df.loc[evolvee].dropna()
    for mutation in row.index:
        mut_tupels.append((mutation, evolvee, row[mutation], full_count_ref_df.loc[evolvee, mutation],
                           input_mutations_df.loc[mutation]["amount"]))

new_df = pd.DataFrame(mut_tupels, columns=["From", "To", "Prevalence", "Amount", "Input amount"])
new_df.to_excel(path_csv)

# THE FIRST ROW IS THE MUTATIONS THAT EVOLVE INTO THE FIRST COLUMN MUTATIONS; so each column is which mutation has the
# row name evolved into
