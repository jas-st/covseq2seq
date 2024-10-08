import torch
import torch.nn as nn
import torch.nn.functional as f
import math
import os
import time
import data_handler2 as dh
from torch import optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, training=True, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)
        self.training = training

    def forward(self, inpt):
        embedded = self.embedding(inpt)
        output, hidden = self.gru(embedded)

        if self.training:
            output = self.dropout(output)
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, m_length, sos_token):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.m_length = m_length
        self.sos_token = sos_token

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(self.sos_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(self.m_length):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = f.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None  # We return `None` for consistency in the training loop

    def forward_step(self, inpt, hidden):
        try:
            output, hidden = self.gru(self.embedding(inpt), hidden)
        except RuntimeError:
            print(inpt)
            print(inpt.shape)
        output = self.out(output)
        return output, hidden


def train_epoch(dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, path):
    total_loss = 0
    encoder.training = True
    os.makedirs(path+'/enc', exist_ok=True)
    os.makedirs(path+'/dec', exist_ok=True)
    tracker = 390 # random number

    print("training...", flush=True)
    for count, data in enumerate(dataloader):
        progress = round(count*100/len(dataloader))
        if tracker != progress:
            print("Data Progress: %d%%" % progress, flush=True)
            tracker = progress
        # \r is carriage return - returns it to the beginning, replacing characters
        # the flush forces the buffer to print everything out
        # sys.stdout.write("Data Progress: %.2f%%   \r" % progress)
        # sys.stdout.flush()
        input_tensor, target_tensor = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()
        print(loss.item())
        total_loss += loss.item()

        torch.save(encoder.state_dict(), path+"/enc/enc.pth")
        torch.save(decoder.state_dict(), path+"/dec/dec.pth")

    return total_loss / len(dataloader)


def asminutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timesince(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (asminutes(s), asminutes(rs))


def train(dataloader, encoder, decoder, n_epochs, path, learning_rate=0.001, print_every=2):
    start = time.time()
    print_loss_total = 0  # Reset every print_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss(ignore_index=0)
    # criterion = nn.CrossEntropyLoss() # used when the outputs are directly the probabilities?

    for epoch in range(1, n_epochs + 1):
        print("Epoch", epoch, flush=True)
        loss = train_epoch(dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, path)
        print_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timesince(start, epoch / n_epochs), epoch,
                                         epoch / n_epochs * 100, print_loss_avg), flush=True)


def decode_ids(ids, translator, end=True):
    words = []
    for id_ in ids:
        if id_.item() == 3 and end:
            break
        elif id_.item() == 2:
            continue
        elif id_.item() == 0:
            continue

        words.append(translator[id_.item()])

    return words


def evol_simulator(inpt, encoder, decoder, evol_list, lang):
    if len(evol_list) > 5:
        return evol_list

    encoder_outputs, encoder_hidden = encoder(inpt)
    decoder_outputs, decoder_hidden, _ = decoder(encoder_outputs, encoder_hidden)
    _, topi = decoder_outputs.topk(1)
    decoded_ids = topi.squeeze()

    decoded_words = decode_ids(decoded_ids, lang.index_word)
    input_words = decode_ids(inpt.squeeze(), lang.index_word)

    if len(decoded_words) != 0:
        new_seq = decoded_words + [x for x in input_words if x != "<OOV>"]
        new_seq.sort(key=lambda x: int(x[1:-1]))
        evol_list.append(new_seq)

        seq_encoded, _ = dh.tokenize([str(dh.preprocess_sentence(" ".join(new_seq)))], lang)
        return evol_simulator(torch.LongTensor(seq_encoded), encoder, decoder,
                              evol_list, lang)

    else:
        evol_list.append(decoded_words)
        return evol_list


def beam_evol(inpt, encoder, decoder, lang):
    encoder_outputs, encoder_hidden = encoder(inpt)
    decoder_outputs, decoder_hidden, _ = decoder(encoder_outputs, encoder_hidden)

    input_words = decode_ids(inpt.squeeze(), lang.index_word)
    _, indexes = torch.topk(decoder_outputs, 5)
    idxs = indexes.squeeze()
    beam1 = idxs[:, 0:1].squeeze()
    beam2 = idxs[:, 1:2].squeeze()
    beam3 = idxs[:, 2:3].squeeze()
    beam4 = idxs[:, 3:4].squeeze()
    beam5 = idxs[:, 4:5].squeeze()

    beam1_decoded = decode_ids(beam1, lang.index_word) + [x for x in input_words if x != "<OOV>"]
    beam2_decoded = decode_ids(beam2, lang.index_word) + [x for x in input_words if x != "<OOV>"]
    beam3_decoded = decode_ids(beam3, lang.index_word) + [x for x in input_words if x != "<OOV>"]
    beam4_decoded = decode_ids(beam4, lang.index_word) + [x for x in input_words if x != "<OOV>"]
    beam5_decoded = decode_ids(beam5, lang.index_word) + [x for x in input_words if x != "<OOV>"]

    return [beam1_decoded, beam2_decoded, beam3_decoded, beam4_decoded, beam5_decoded]
