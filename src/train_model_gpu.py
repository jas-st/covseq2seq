import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import models
import pickle
import os

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_size = 256
batch_size = 1
path = "./educational"
epochs = 10

if torch.cuda.is_available():
    print("Using GPU")
else:
    print("Using CPU")


inpt_tensor = torch.load("./data/inpt_tensor2")
tgt_tensor = torch.load("./data/tgt_tensor2")
with open('./data/data_lang_full.pkl', "rb") as mut:
    data_lang_full = pickle.load(mut)

max_output = len(tgt_tensor.__getitem__(0))
sos_token = data_lang_full.word_index["<start>"]
vocab_length = len(data_lang_full.word_index)
print("Vocab:", vocab_length)

# train_data = TensorDataset(torch.LongTensor(inpt_tensor).to(device),
#                            torch.LongTensor(tgt_tensor).to(device))
train_data = TensorDataset(torch.LongTensor(inpt_tensor[0]).to(device)[None, :],
                           torch.LongTensor(tgt_tensor[0]).to(device)[None, :])
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

enc = models.EncoderRNN(vocab_length+1, hidden_size).to(device)
dec = models.DecoderRNN(hidden_size, vocab_length+1, max_output, sos_token).to(device)


models.train(train_dataloader, enc, dec, epochs, path, print_every=1)