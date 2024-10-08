import torch
from sklearn.model_selection import train_test_split
import pickle
import data_handler2 as dh


with open('data/data_input.pkl', "rb") as mut:
    inp_train = pickle.load(mut)

with open('data/data_target.pkl', "rb") as mut:
    tgt_train = pickle.load(mut)

inp_lang = [dh.preprocess_sentence(x) for x in inp_train]
tgt_lang = [dh.preprocess_sentence(x) for x in tgt_train]

full_set = inp_lang + tgt_lang

_, data_lang = dh.tokenize(full_set)

# with open("data_lang_full.pkl", "wb") as fo:
#     pickle.dump(data_lang, fo, protocol=pickle.HIGHEST_PROTOCOL)

inp, inp_t, tgt, tgt_t = train_test_split(inp_lang, tgt_lang, train_size=0.85)

inpt_tensor, _ = dh.tokenize(inp, data_lang)
tgt_tensor, _ = dh.tokenize(tgt, data_lang)

inpt_tensor_t, _ = dh.tokenize(inp_t, data_lang)
tgt_tensor_t, _ = dh.tokenize(tgt_t, data_lang)

data_lang_dict = data_lang.word_index
data_lang_reverse = data_lang.index_word

# torch.save(inpt_tensor, "inpt_tensor2")
# torch.save(tgt_tensor, "tgt_tensor2")
#
# torch.save(inpt_tensor_t, "inpt_tensor_train2")
# torch.save(tgt_tensor_t, "tgt_tensor_train2")
#
# with open("data_lang2.pkl", "wb") as fo:
#     pickle.dump(data_lang_dict, fo, protocol=pickle.HIGHEST_PROTOCOL)
#
# with open("data_lang_reverse2.pkl", "wb") as fo:
#     pickle.dump(data_lang_reverse, fo, protocol=pickle.HIGHEST_PROTOCOL)

