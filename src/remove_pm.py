import pandas as pd
import data_loader as dl
import data_handler2 as dh

# path = "C:\\Users\\zhasmi00\\Downloads\\2022\\August2022\\gisaid_KW29_August2022.csv"
path = "C:\\Users\\zhasmi00\\Downloads\\test\\RESULTS\\AugustKW29_22_beam\\evolutions_ausgewertet_2.csv"
save_path = "C:\\Users\\zhasmi00\\Downloads\\test\\RESULTS\\AugustKW29_22_beam\\gisaid_KW29_August2022_removed_2.fasta"
ref_seq = [x for x in dl.REFSEQ]

df = pd.read_csv(path, sep=';', index_col="seqName", usecols=["seqName", "partiallyAliased", "substitutions",
                                                              "qc.privateMutations.status",
                                                              "privateNucMutations.unlabeledSubstitutions"])
df = df[~df.index.duplicated(keep='first')]

for index in df.index:
    if df.loc[index, 'qc.privateMutations.status'] != 'good':
        row = df.loc[index]

        try:
            subs = [x for x in row["substitutions"].split(",") if x not in
                    row["privateNucMutations.unlabeledSubstitutions"].split(",")]
        except AttributeError:
            continue

        input_string = dh.create_fasta(ref_seq, subs)

        with open(save_path, 'a') as file:
            fastastring = ">%s\n%s\n" % (index, input_string)
            file.write(fastastring)


# so basically you create a new fasta file for only those that arent good, then you auswerrten them using nextclade
# and then you join merge whatev the new data frame, leaving only those that overlap, essentially leaving those that
# are mediocre - but thats in a new script?




