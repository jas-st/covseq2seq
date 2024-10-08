import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import models
import pickle
import data_loader as dl
import data_handler2 as dh

save_path = "C:\\Users\\zhasmi00\\Downloads\\test\\RESULTS\\AugustKW29_22_beam\\"
tensor_path = "test_tensor_august_3"
fasta_path = "evolutionsKW29august_beam_3.fasta"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_size = 256
batch_size = 1
ref_seq = [x for x in dl.REFSEQ]

inpt_tensor = torch.load(save_path + tensor_path)
tgt_tensor = torch.load("./data/tgt_tensor_train2")

with open('./data/data_lang_full.pkl', "rb") as mut:
    data_lang = pickle.load(mut)


max_output = len(tgt_tensor.__getitem__(0))
sos_token = data_lang.word_index["<start>"]
vocab_length = len(data_lang.word_index)

train_data = TensorDataset(torch.LongTensor(inpt_tensor).to(device))
                          # torch.LongTensor(tgt_tensor).to(device))
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

enc = models.EncoderRNN(vocab_length+1, hidden_size).to(device)
dec = models.DecoderRNN(hidden_size, vocab_length+1, max_output, sos_token).to(device)


enc.load_state_dict(torch.load("./models2/enc/enc.pth", map_location=torch.device("cpu")))
dec.load_state_dict(torch.load("./models2/dec/dec.pth", map_location=torch.device("cpu")))
enc.eval()
dec.eval()
idx = 1

for data in train_dataloader:
    seq_input = data[0]

    evolution = models.beam_evol(seq_input, enc, dec, data_lang)

    evol_list = []
    inpt_seq = models.decode_ids(seq_input.squeeze(), data_lang.index_word)
    input_string = dh.add_mutations(ref_seq, inpt_seq)

    for evol in evolution:
        evol_list.append(dh.add_mutations(ref_seq, evol))

    with open(save_path + fasta_path, 'a') as file:
        fastastring = ">input_%s\n%s\n" % (idx, input_string)
        file.write(fastastring)
        for j in range(len(evol_list)):
            fastastring = ">%s\n%s\n" % (f"beam{j+1}_{idx}", evol_list[j])
            file.write(fastastring)
        idx += 1


