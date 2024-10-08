import data_loader as dl
import pandas as pd
import data_handler2 as dh

ref_seq = [x for x in dl.REFSEQ]
path_df = "C:\\Users\\zhasmi00\\Downloads\\test\\august\\evols2022lineages_cleaned_2.xlsx"
save_path = "C:\\Users\\zhasmi00\\Downloads\\test\\august\\evols2022lineages_2.fasta"
df = pd.read_excel(path_df)
df.drop_duplicates(subset=["substitutions"], inplace=True)


with open(save_path, 'w') as file:

    for idx in df.index:
        row = df.loc[idx]

        seqID = row["seqName"]
        try:
            mutations = row["substitutions"].split(",")
        except AttributeError:
            mutations = []

        seq = dh.create_fasta(ref_seq, mutations)

        fastastring = ">%s\n%s\n" % (seqID, seq)
        file.write(fastastring)

