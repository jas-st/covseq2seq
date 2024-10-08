import pandas as pd
import pickle

path_database = "C:\\Users\\zhasmi00\\PycharmProjects\\mutation_database\\"
path_lineages = path_database + "distribution_lineages.txt"
path_summary = path_database + "Properties_values.csv"

df_lineages = pd.read_csv(path_lineages, sep="\t", usecols=["Mutation ID", "Mutated sequences count",
                                                            "Frequency of mutated sequences per Lineage", "Lineage"])
df_summary = pd.read_csv(path_summary, usecols=["Mutation ID", "Total_changes"])

with open("./id_dict_reverse.pkl", "rb") as fo:
    id_dict_reverse = pickle.load(fo)

with open("./id_dict.pkl", "rb") as fo:
    id_dict_true = pickle.load(fo)

with open("./set_dict_05.pkl", "rb") as fo:
    set_05 = pickle.load(fo)


# accepts THE IDS of mutations
def get_shared_lineages(mutation1, mutation2, ref_df=df_lineages, desired_freq=0.5):
    df = ref_df[(ref_df["Mutation ID"].isin([mutation1, mutation2])) &
                (ref_df["Frequency of mutated sequences per Lineage"] > desired_freq)]

    df = df[df.duplicated(subset=["Lineage"], keep=False)]["Lineage"].unique()

    return df, len(df)


def check_adjacent_mutations(mutation_ref, mutation_compare, id_dict=id_dict_reverse,
                             ref_df=df_lineages, desired_freq=0):
    # MUTATION REF IS IN A254G FORMAT; COMPARE IS IN ID FORMAT
    # get allows you to provide default value if key is missing
    if mutation_ref is None:
        return 0
    get_dict = id_dict.get(int(mutation_ref[1:-1]), None)

    if get_dict is None or len(get_dict) == 1:
        return 0

    amount = 0
    for variation in get_dict.keys():
        if variation != mutation_ref:
            amount += get_shared_lineages(mutation_compare, get_dict[variation], ref_df=ref_df,
                                          desired_freq=desired_freq)[1]

    return amount


def get_parameters(mutation):
    score = 0
    if mutation is not None:
        score = df_summary[df_summary["Mutation ID"] == mutation]["Total_changes"].values[0]
    return score


def get_reward(mutation_ref, mutation_ref_nuc, mutation_compare, param_values):
    return get_shared_lineages(mutation_ref, mutation_compare)[1] \
           + check_adjacent_mutations(mutation_ref_nuc, mutation_compare) \
           + param_values


def check_mutation_freq(mut_id, ref_df=df_lineages, desired_freq=0.5):
    df = ref_df[(ref_df["Mutation ID"] == mut_id) &
                (ref_df["Frequency of mutated sequences per Lineage"] > desired_freq)]

    df = df["Lineage"].unique()

    return set(df)


def create_mut_freq_count_dict(set_name, desired_freq=0.5):
    set_dict = {}

    for key in id_dict_true.keys():
        set_dict[key] = check_mutation_freq(key, desired_freq=desired_freq)

    with open(set_name + ".pkl", "wb") as file:
        pickle.dump(set_dict, file, pickle.HIGHEST_PROTOCOL)


def get_interaction_dict():
    interaction_dict = {}
    keys = list(id_dict_true.keys())

    for prog, val1 in enumerate(keys):
        set_val1 = set_05[val1]
        if set_val1 == 0:
            continue

        for val2_n in range(prog+1, len(keys)):
            val2 = keys[val2_n]
            set_val2 = set_05[val2]
            if set_val2 == 0:
                continue

            result = len(set_val1.intersection(set_val2))
            if result != 0:
                interaction_dict[frozenset([val1, val2])] = result
                print(val1, val2, result)

    # with open("interaction_dict.pkl", "wb") as file:
    #     pickle.dump(interaction_dict, file, pickle.HIGHEST_PROTOCOL)


'''
It would be fairly straightforward to write a subclass of dict that took iterables as 
key arguments and then turned then into a frozenset when storing values:

class SymDict(dict):
    def __getitem__(self, key):
        return dict.__getitem__(self, frozenset(key))

    def __setitem__(self, key, value):
        dict.__setitem__(self, frozenset(key), value)
'''