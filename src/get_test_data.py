import pandas as pd
import pickle
import utils
import data_handler2 as dh
import torch
import os

# path = "C:\\Users\\zhasmi00\\Downloads\\test\\nextclade.csv"
path = "C:\\Users\\zhasmi00\\Downloads\\test\\RESULTS\\AugustKW29_22_beam\\evolutions_ausgewertet_3_cleaned.csv"
save_path = "C:\\Users\\zhasmi00\\Downloads\\test\\RESULTS\\AugustKW29_22_beam\\"
tensor_name = "test_tensor0"
os.makedirs(save_path, exist_ok=True)

df_ref = pd.read_csv(path, sep=';', index_col="seqName", usecols=["seqName", "partiallyAliased", "qc.overallStatus","substitutions"])
                                                                  #"qc.privateMutations.status", "substitutions",
                                                                  #"privateNucMutations.unlabeledSubstitutions"])

# remove duplicates
df_ref = df_ref[~df_ref.index.duplicated(keep='first')]

# create ref dict
ref_dict = utils.create_dictionary(df_ref)

data_list = []

for key in ref_dict:
    l1 = ref_dict[key]

    for seq in l1:
        data_list.append(" ".join(seq))

dataset = [dh.preprocess_sentence(x) for x in data_list]
print(len(dataset))

with open("data/data_lang_full.pkl", "rb") as fo:
    data_lang = pickle.load(fo)

inpt_tensor, _ = dh.tokenize(dataset, data_lang)

torch.save(inpt_tensor, save_path + tensor_name)
