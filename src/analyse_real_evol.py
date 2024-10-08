import pickle
import pandas as pd

path1 = "./id_nuc_dict.pkl"
path2 = "./prot_id_dict.pkl"
path3 = "./variant_gra.csv"

with open(path1, "rb") as fo:
    id_nuc = pickle.load(fo)

with open(path2, "rb") as fo:
    prot_id = pickle.load(fo)
del prot_id[list(prot_id.keys())[0]]

# df = pd.read_csv(path3)

print([x for x in prot_id.keys() if "L452" in x and len(x) == 5])
print(prot_id["L452Q"])
print(id_nuc[prot_id["L452R"]])





