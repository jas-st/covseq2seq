import utils
import pickle

path = "C:\\Users\\zhasmi00\\Downloads"
data_list = utils.get_data(path, years=["2021"])[20:24]
year = "2021"
print("Using the following data: ", data_list)


path1 = "C:\\Users\\zhasmi00\\Downloads\\" + year + '\\' + data_list[0].split('_')[-1][:-4] + '\\' + data_list[0]
path2 = "C:\\Users\\zhasmi00\\Downloads\\" + year + '\\' + data_list[1].split('_')[-1][:-4] + '\\' + data_list[1]
path3 = "C:\\Users\\zhasmi00\\Downloads\\" + year + '\\' + data_list[2].split('_')[-1][:-4] + '\\' + data_list[2]
path4 = "C:\\Users\\zhasmi00\\Downloads\\" + year + '\\' + data_list[3].split('_')[-1][:-4] + '\\' + data_list[3]

save_folder = "RESULTS\\JuneKW21_21"

save_path1 = "C:\\Users\\zhasmi00\\Downloads\\test\\" + save_folder + "\\unobserved_lins.pkl"
save_path2 = "C:\\Users\\zhasmi00\\Downloads\\test\\" + save_folder + "\\old_lins.pkl"
save_path3 = "C:\\Users\\zhasmi00\\Downloads\\test\\" + save_folder + "\\new_lins.pkl"

# NEW LINEAGES (UNSEEN IN THE OLD WEEK)
# OBSERVED EVOLUTIONS

# create ref dicts
ref_dict_old = utils.create_ref_dict(path1)
ref_dict_new1 = utils.create_ref_dict(path2)
ref_dict_new2 = utils.create_ref_dict(path3)
ref_dict_new3 = utils.create_ref_dict(path4)

unobserved_lins1 = set([x for x in ref_dict_new1.keys() if x not in ref_dict_old.keys()])
unobserved_lins2 = set([x for x in ref_dict_new2.keys() if x not in ref_dict_old.keys()])
unobserved_lins3 = set([x for x in ref_dict_new3.keys() if x not in ref_dict_old.keys()])

unobserved_lins = unobserved_lins1.union(unobserved_lins3, unobserved_lins2)

next_week_lins = set(list(ref_dict_new1.keys()) + list(ref_dict_new2.keys()) + list(ref_dict_new3.keys()))

# with open("observations.txt", 'w') as file:
#     file.write("NEW LINEAGES WEEK 1\n")
#     file.write(", ".join(unobserved_lins1))
#     file.write("\n\n")
#
#     file.write("NEW LINEAGES WEEK 2\n")
#     file.write(", ".join(unobserved_lins2))
#     file.write("\n\n")
#
#     file.write("NEW LINEAGES WEEK 3\n")
#     file.write(", ".join(unobserved_lins3))
#     file.write("\n\n")

with open(save_path1, "wb") as fo:
    pickle.dump(list(unobserved_lins), fo, protocol=pickle.HIGHEST_PROTOCOL)

with open(save_path2, "wb") as fo:
    pickle.dump(list(ref_dict_old.keys()), fo, protocol=pickle.HIGHEST_PROTOCOL)

with open(save_path3, "wb") as fo:
    pickle.dump(next_week_lins, fo, protocol=pickle.HIGHEST_PROTOCOL)
