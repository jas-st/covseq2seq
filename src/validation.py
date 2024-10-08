import pandas as pd
import pickle
import utils
from collections import defaultdict

save_folder = "RESULTS\\JuneKW21_21"

# unobserved lineges in the actual weeks vs the input week
path_unobserved = "C:\\Users\\zhasmi00\\Downloads\\test\\" + save_folder + "\\unobserved_lins.pkl"
# lineages in the old week
path_old = "C:\\Users\\zhasmi00\\Downloads\\test\\" + save_folder + "\\old_lins.pkl"
# lineages in the new weeks
path_new = "C:\\Users\\zhasmi00\\Downloads\\test\\" + save_folder + "\\new_lins.pkl"
# result data frame
path_res = "C:\\Users\\zhasmi00\\Downloads\\test\\" + save_folder + "\\evolutions_June21_evaluated.csv"
# path to save the validation
path_val = "C:\\Users\\zhasmi00\\Downloads\\test\\" + save_folder + "\\validation.txt"


path = "C:\\Users\\zhasmi00\\Downloads"
data_list = utils.get_data(path, years=["2021"])[20:24]
year = "2021"
print("Using the following data: ", data_list)

# input data frame
input_path = "C:\\Users\\zhasmi00\\Downloads\\" + year + '\\' + data_list[0].split('_')[-1][:-4] + '\\' + data_list[0]
# next 3 weeks
path_df1 = "C:\\Users\\zhasmi00\\Downloads\\" + year + '\\' + data_list[1].split('_')[-1][:-4] + '\\' + data_list[1]
path_df2 = "C:\\Users\\zhasmi00\\Downloads\\" + year + '\\' + data_list[2].split('_')[-1][:-4] + '\\' + data_list[2]
path_df3 = "C:\\Users\\zhasmi00\\Downloads\\" + year + '\\' + data_list[3].split('_')[-1][:-4] + '\\' + data_list[3]

# every observed training set
with open('lineage_set.pkl', "rb") as mut:
    lineage_set = pickle.load(mut)

with open(path_old, "rb") as mut:
    old_lins = pickle.load(mut)

with open(path_unobserved, "rb") as mut:
    new_lineages_list = pickle.load(mut)

with open(path_new, "rb") as mut:
    new_lins = pickle.load(mut)

df_res = pd.read_csv(path_res, sep=';', usecols=["seqName", "partiallyAliased", "qc.overallStatus"])
df_res["evolID"] = df_res["seqName"].str.split("_", expand=True)[1]

df_without_input = df_res.drop_duplicates(["partiallyAliased", "evolID"])
df_without_input = df_without_input.drop(df_without_input[df_without_input['qc.overallStatus'] == 'bad'].index)
df_without_input = df_without_input[~df_without_input["seqName"].str.contains("input")]

df_res = df_res.drop(df_res[df_res['qc.overallStatus'] == 'bad'].index)

result_lins = df_without_input["partiallyAliased"].values

with open(path_val, 'w') as file:
    file.write("VALIDATION\n\n")


# Novel lineages that model hasn't trained on
novel_lins = set([x for x in result_lins if x not in lineage_set and len(x)>1])

with open(path_val, 'a') as file:
    file.write("Novel lineages that the model hasn't trained on \n")
    file.write(", ".join(novel_lins))
    file.write("\n\n")

# Lineages that haven't been seen in the input week
unobserved_new_lins = set([x for x in result_lins if x in new_lineages_list])

with open(path_val, 'a') as file:
    file.write("New lineages unseen in the input week but seen in the next weeks\n")
    file.write(", ".join(unobserved_new_lins))
    file.write("\n\n")

# Lineages seen in the next weeks but not seen in the results
unpredicted_lins = set([x for x in new_lins if x not in result_lins and len(x)>3 and x not in old_lins])

with open(path_val, 'a') as file:
    file.write("Lineages seen in real weeks, but not predicted\n")
    file.write(", ".join(unpredicted_lins))
    file.write("\n\n")

# Top Lineages
# for now just count every lineage in the two weeks and sort by prevalence
# for the results df: count the distinct evolutions minus the input
input_counter = defaultdict(lambda: 0)

df_old1_counter = defaultdict(lambda: 0)
df_old2_counter = defaultdict(lambda: 0)
df_old3_counter = defaultdict(lambda: 0)
df_res1_counter = defaultdict(lambda: 0)
df_res2_counter = defaultdict(lambda: 0)
df_res3_counter = defaultdict(lambda: 0)
df_res4_counter = defaultdict(lambda: 0)
df_res5_counter = defaultdict(lambda: 0)
df_res6_counter = defaultdict(lambda: 0)

input_df = pd.read_csv(input_path, sep=';', index_col="seqName",
                       usecols=["seqName", "partiallyAliased"])
df_ref_old1 = pd.read_csv(path_df1, sep=';', index_col="seqName",
                          usecols=["seqName", "partiallyAliased"])
df_ref_old2 = pd.read_csv(path_df2, sep=';', index_col="seqName",
                          usecols=["seqName", "partiallyAliased"])
df_ref_old3 = pd.read_csv(path_df3, sep=';', index_col="seqName",
                          usecols=["seqName", "partiallyAliased"])

# remove duplicates
input_df = input_df[~input_df.index.duplicated(keep='first')]
df_ref_old1 = df_ref_old1[~df_ref_old1.index.duplicated(keep='first')]
df_ref_old2 = df_ref_old2[~df_ref_old2.index.duplicated(keep='first')]
df_ref_old3 = df_ref_old3[~df_ref_old3.index.duplicated(keep='first')]

for lin in input_df["partiallyAliased"].values:
    input_counter[lin] += 1

for lin in df_ref_old1["partiallyAliased"].values:
    df_old1_counter[lin] += 1

for lin in df_ref_old2["partiallyAliased"].values:
    df_old2_counter[lin] += 1

for lin in df_ref_old3["partiallyAliased"].values:
    df_old3_counter[lin] += 1

for name in df_res.index:
    row = df_res.loc[name]

    if "evol0" in row["seqName"]:
        df_res1_counter[row["partiallyAliased"]] += 1
    elif "evol1" in row["seqName"]:
        df_res2_counter[row["partiallyAliased"]] += 1
    elif "evol2" in row["seqName"]:
        df_res3_counter[row["partiallyAliased"]] += 1
    elif "evol3" in row["seqName"]:
        df_res4_counter[row["partiallyAliased"]] += 1
    elif "evol4" in row["seqName"]:
        df_res5_counter[row["partiallyAliased"]] += 1
    elif "evol5" in row["seqName"]:
        df_res6_counter[row["partiallyAliased"]] += 1


inputs = sorted(input_counter.items(), key=lambda x: x[1], reverse=True)
old1 = sorted(df_old1_counter.items(), key=lambda x: x[1], reverse=True)
old2 = sorted(df_old2_counter.items(), key=lambda x: x[1], reverse=True)
old3 = sorted(df_old3_counter.items(), key=lambda x: x[1], reverse=True)
res1 = sorted(df_res1_counter.items(), key=lambda x: x[1], reverse=True)
res2 = sorted(df_res2_counter.items(), key=lambda x: x[1], reverse=True)
res3 = sorted(df_res3_counter.items(), key=lambda x: x[1], reverse=True)
res4 = sorted(df_res4_counter.items(), key=lambda x: x[1], reverse=True)
res5 = sorted(df_res5_counter.items(), key=lambda x: x[1], reverse=True)
res6 = sorted(df_res6_counter.items(), key=lambda x: x[1], reverse=True)

actual_input = [x[0] for x in inputs[:10]]

actual_top1 = [x[0] for x in old1[:10]]
result_top1 = [x[0] for x in res1[:10]]

actual_top2 = [x[0] for x in old2[:10]]
result_top2 = [x[0] for x in res2[:10]]

actual_top3 = [x[0] for x in old3[:10]]
result_top3 = [x[0] for x in res3[:10]]

result_top4 = [x[0] for x in res4[:10]]
result_top5 = [x[0] for x in res5[:10]]
result_top6 = [x[0] for x in res6[:10]]

with open(path_val, 'a') as file:
    file.write("COMPARISON\n")
    str1 = "Input Week"
    str2 = "Week 1"
    str3 = "Week 2"
    str4 = "Week 3 \n"
    file.write(str1 + " "*(21-len(str1)) + str2 + " "*(21-len(str2)) + str3 + " "*(21-len(str3)) + str4)

    for i in range(10):
        blanks1 = 21 - len(actual_input[i])
        blanks2 = 21 - len(actual_top1[i])
        blanks3 = 21 - len(actual_top2[i])
        file.write(actual_input[i] + " "*blanks1 + actual_top1[i] + " "*blanks2 + actual_top2[i] +
                   " "*blanks3 + actual_top3[i] + "\n")

    file.write("\nMovement: ")
    mov_actual = [x for x in actual_top1+actual_top2+actual_top3 if x not in actual_input]
    file.write(", ".join(set(mov_actual)))
    file.write("\n\n")

    str1 = "Results Step 0"
    str2 = "Results Step 1"
    str3 = "Results Step 2"
    str4 = "Results Step 3"
    str5 = "Results Step 4"
    str6 = "Results Step 5 \n"
    file.write(str1 + " " * (21 - len(str1)) + str2 + " " * (21 - len(str2)) + str3 + " " * (21 - len(str3)) +
               str4 + " " * (21 - len(str4)) + str5 + " " * (21 - len(str5)) + str6)

    for i in range(10):
        blanks1 = 21 - len(result_top1[i])
        blanks2 = 21 - len(result_top2[i])
        blanks3 = 21 - len(result_top3[i])
        blanks4 = 21 - len(result_top4[i])
        blanks5 = 21 - len(result_top5[i])
        file.write(result_top1[i] + " " * blanks1 + result_top2[i] + " " * blanks2 + result_top3[i] +
                   " " * blanks3 + result_top4[i] + " " * blanks4 + result_top5[i] +
                   " " * blanks5 + result_top6[i] + "\n")

    file.write("\nMovement: ")
    mov_result = [x for x in result_top1+result_top2+result_top3+result_top4+result_top5+result_top6
                  if x not in actual_input]
    file.write(", ".join(set(mov_result)))

    file.write("\n\nMatching movement: \n")
    file.write(", ".join(set([x for x in mov_actual if x in mov_result])))
