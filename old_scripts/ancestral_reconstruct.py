from Bio import Phylo
import pandas as pd
from collections import defaultdict
import pickle


def get_parent(tr, child_clade):
    node_path = tr.get_path(child_clade)
    return node_path[-2]


# chunks = 29903
# ancestral_variants = {}

# read in dataframe by chunks
# for chunk in pd.read_table("aligned_seqs.fasta.state", skiprows=8, chunksize=chunks):
#     ancestral_variants[chunk.Node.values[0]] = "".join(chunk.State.values)
#
# with open("reconstructed_ancestors.fasta", 'w') as fp:
#     for node in ancestral_variants:
#         fasta_string = ">" + node + "\n" + ancestral_variants[node] + "\n"
#         fp.write(fasta_string)

treefile = "aligned_seqs.fasta.treefile"
tree = Phylo.read(treefile, "newick")
train_pairs = defaultdict(set)

for clade in tree.find_clades():
    # print("Clade:", clade)
    try:
        train_pairs[get_parent(tree, clade).name].add(clade.name)
    except IndexError:
        continue

with open("train_pairs.json", "wb") as fo:
    pickle.dump(train_pairs, fo, protocol=pickle.HIGHEST_PROTOCOL)
