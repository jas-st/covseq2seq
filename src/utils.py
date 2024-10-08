import itertools
from random import randint
from natsort import natsorted
import pandas as pd
import os
from collections import defaultdict

MONTHS = ["Jan", "Feb", "March", "April", "May", "June", "July", "August", "Sep", "Oct", "Nov", "Dec"]
YEARS = ["2020", "2021", "2022"]
NUCL_MAP = ['A', 'C', 'G', 'T']
GENE_MAP = ['ORF1a', 'ORF1b', 'ORF3a', 'S', 'E', 'M', 'ORF6', 'ORF7a', 'ORF7b', 'ORF8', 'N', 'ORF10', 'Empty']


def get_data(root_name, trim=None, years=None, months=None):
    if months is None:
        months = MONTHS
    if years is None:
        years = YEARS

    m_combos = itertools.product(years, months)
    time_list = []
    for ele in m_combos:
        time_list.append("".join([ele[1], ele[0]]))

    if trim is not None:
        time_list = time_list[:trim]
    df_list = []

    for count, month in enumerate(time_list):
        year = month[-4:]
        kw_dfs = []
        path = root_name + '\\' + year + '\\' + month

        for file in os.listdir(path):
            if file.endswith('.csv'):
                kw_dfs.append(file)

        kw_dfs = natsorted(kw_dfs)
        df_list += kw_dfs

    return df_list


def find_gene(pos_n, gene_map):
    for gene in gene_map:
        _, ranges_start, ranges_end = gene_map[gene][0]
        if ranges_start < pos_n < ranges_end:
            return gene


def get_frequencies(amnt, total_count):
    return round(amnt / total_count, 5)


def get_growth(indx, refdict):
    return refdict[indx]


def handle_non_numerical_data(df):
    columns = ['reference', 'mutated', 'prev_1', 'prev_2', 'prev_3', 'next_1', 'next_2', 'next_3', 'gene']

    for column in columns:

        def convert_to_int(val):
            if gene:
                return GENE_MAP[val]
            else:
                return NUCL_MAP[val]

        if column == "gene":
            gene = True
            df[column] = list(map(convert_to_int, df[column]))
        else:
            gene = False
            df[column] = list(map(convert_to_int, df[column]))

    return df


def onehot_encode(dataframe):
    columns = ['reference', 'mutated', 'prev_1', 'prev_2', 'prev_3', 'next_1', 'next_2', 'next_3', 'gene']

    for column in columns:
        if column != "gene":
            dataframe = dataframe.join(pd.get_dummies(dataframe[column].astype(
                pd.CategoricalDtype(categories=NUCL_MAP)), prefix=column))
            dataframe.drop(column, axis=1, inplace=True)
        else:
            dataframe = dataframe.join(pd.get_dummies(dataframe[column].astype(
                pd.CategoricalDtype(categories=GENE_MAP))))
            dataframe.drop(column, axis=1, inplace=True)

    ind_freq = dataframe.pop("ind_freq")
    dataframe.insert(len(dataframe.columns), "ind_freq", ind_freq)
    return dataframe


# def find_aa_change(position, gene, aa_positions, aa_changes):
#     change = "No_AA_change"
#
#     if len(aa_positions) != "0":
#         if gene is not None and gene in aa_positions:
#             aa_mut = pos.na2aa_genome(position, gene)
#             if str(aa_mut) in aa_positions[gene]:
#                 change = aa_changes[gene][str(aa_mut)]
#
#     return change


def percent_categories(value):
    if value < 0.01:
        return 0
    else:
        return int(value * 10) + 1


# corr = DATA.corr()
# cor_target = abs(corr['ind_freq_next'])
# #Selecting highly correlated features
# relevant_features = cor_target[cor_target > 0.5]
# print(relevant_features)

def create_dictionary(df):
    df_dict = defaultdict(list)

    for name in df.index:
        # remove those samples where the status isnt good or too many private mutations
        # if df.loc[name, 'qc.overallStatus'] != "good" or df.loc[name, 'qc.privateMutations.status'] != 'good':
        #     continue

        row = df.loc[name]
        lin = row["partiallyAliased"]

        if type(row["substitutions"]) != float:
            subs = row["substitutions"].split(",")
        else:
            continue

        # if type(row["privateNucMutations.unlabeledSubstitutions"]) != float:
        #     private_muts = row["privateNucMutations.unlabeledSubstitutions"].split(",")
        # else:
        #     private_muts = []

        subs_reduced = subs

        # subs_reduced = [x for x in subs if x not in private_muts]
        # if len(subs_reduced) == 0:
        #     continue

        subs_reduced.sort(key=lambda x: int(x[1:-1]))

        if subs_reduced not in df_dict[lin]:
            df_dict[lin].append(subs_reduced)

    return df_dict


def create_pairs(list1, list2):
    pair_list = []

    for l2 in list2:
        l1 = list1[randint(0, len(list1) - 1)]
        new_l2 = [x for x in l2 if x not in l1]
        if len(new_l2) > len(l1):
            continue

        pair_list.append((" ".join(l1), " ".join(new_l2)))

    return pair_list


def create_ref_dict(path):
    df = pd.read_csv(path, sep=';', index_col="seqName",
                     usecols=["seqName", "partiallyAliased", "qc.overallStatus",
                              "qc.privateMutations.status", "substitutions",
                              "privateNucMutations.unlabeledSubstitutions"])

    # remove duplicates
    df = df[~df.index.duplicated(keep='first')]

    # create ref dict
    ref_dict = create_dictionary(df)

    return ref_dict


def get_hex():
    random_number = randint(0, 16777215)  # max hex color is FFFFFF, which corresponds to 16^6-1
    hex_number = str(hex(random_number))
    hex_color = '#' + hex_number[2:]  # cuz it starts with some prefix probably

    return hex_color


def get_hex_unique(used_colors):
    hex_color = get_hex()

    while hex_color in used_colors:
        hex_color = get_hex()

    return hex_color


