import pandas as pd


def status_function(row, effects=("Deceased", "Hospitalized")):
    # counts the True values in the data frame
    row = row[row["Effect"].isin(effects)]
    return (row["P value"] < 0.001).sum()


def physicochem_function(row):
    ref_values = df_pchemical_ref[df_pchemical_ref["Gene ID"] == row["Gene ID"].values[0]]
    diff_df = pd.concat([ref_values, row]).drop(["Gene ID", "Mutation ID"], axis=1)
    changes_series = (diff_df.iloc[0] - diff_df.iloc[1]).abs() > diff_df.iloc[0].abs()*0.1

    return changes_series.sum()/5


def protein_function(row):
    if row["Score"].values[0] >= 0.5:
        return 1
    else:
        return 0


def antig_immun(row):
    changes = row[["Significant changes in antigenicity", "Significant changes in immunogenicity"]] == "yes"
    return changes.sum(axis=1)


def binding(row):
    if row["Delta ave bind"].values[0] > 0 and row["P-value"].values[0] < 0.05:
        return 1
    else:
        return 0


path_database = "C:\\Users\\zhasmi00\\PycharmProjects\\mutation_database\\"

path_antig_immun = path_database + "antigenicity_immunogenicity.txt"
path_status = path_database + "patient_status.txt"
path_pchemical = path_database + "physicochemical_prop.txt"
path_pchemical_ref = path_database + "physicochemical_reference_val.txt"
path_prot_function = path_database + "protein_function.txt"
path_binding = path_database + "rbd_binding_ace2.txt"

df_antig_immun = pd.read_csv(path_antig_immun, sep="\t", usecols=["Mutation ID", "Significant changes in antigenicity",
                                                                  "Significant changes in immunogenicity"])
df_status = pd.read_csv(path_status, sep="\t", usecols=["Mutation ID", "Effect", "P value", "Direction"])
df_pchemical = pd.read_csv(path_pchemical, sep="\t", usecols=["Mutation ID", "Gene ID", "Molecular weight",
                                                              "Theoretical PI", "Extinction coefficients",
                                                              "Aliphatic index", "grand average of hydropathicity"])
df_pchemical_ref = pd.read_csv(path_pchemical_ref, sep="\t", usecols=["Gene ID", "Molecular weight", "Theoretical PI",
                                                                      "Extinction coefficients", "Aliphatic index",
                                                                      "grand average of hydropathicity"])
df_prot_function = pd.read_csv(path_prot_function, sep="\t", usecols=["Mutation ID",  "Score"])
df_binding = pd.read_csv(path_binding, sep="\t", usecols=["Mutation ID", "Delta ave bind", "P-value"])

df_merged = df_status.groupby("Mutation ID").apply(lambda x: status_function(x)).to_frame(name="Status")
df_merged["Physicochem_changes"] = df_pchemical.groupby("Mutation ID").apply(lambda x: physicochem_function(x))
df_merged["Protein_function"] = df_prot_function.groupby("Mutation ID").apply(lambda x: protein_function(x))
df_merged["Antigenicity_immunogenicity"] = df_antig_immun.groupby("Mutation ID").apply(lambda x: antig_immun(x))
df_merged["ACE2_Binding"] = df_binding.groupby("Mutation ID").apply(lambda x: binding(x))

df_merged.fillna(0, inplace=True)

df_merged["Total_changes"] = df_merged.sum(axis=1)
df_merged.to_csv("Properties_values.csv")
