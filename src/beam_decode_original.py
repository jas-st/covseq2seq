import torch
from queue import PriorityQueue
from itertools import count
import operator
import pickle

import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_size = 256
batch_size = 1
SOS_token = 2
EOS_token = 3
other_token = 0

with open("data/data_lang_full.pkl", "rb") as fo:
    data_lang = pickle.load(fo)

with open("./corr_dict.pkl", "rb") as fo:
    data_corr = pickle.load(fo)

with open("./id_nuc_dict.pkl", "rb") as fo:
    id_nuc_rl = pickle.load(fo)

id_nuc = {k: v for (v, k) in id_nuc_rl.items()}


class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length, seq, corr):
        """
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        """
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length
        self.seq = seq
        self.corr = corr

    def eval_corr(self, start_seq):
        reward_corr = 0
        curr_node = self.seq[-1]

        if curr_node not in data_corr:
            return reward_corr

        for mut in start_seq:
            if mut in data_corr[curr_node]:# and data_corr[curr_node][mut] > 0:
                reward_corr += data_corr[curr_node][mut]

        for mut in self.seq:
            if mut in data_corr[curr_node]:# and data_corr[curr_node][mut] > 0:
                reward_corr += data_corr[curr_node][mut]

        # print(curr_node, reward_corr)
        # print(self.seq)
        return reward_corr

    def eval(self, start_seq):
        if self.wordid not in [2, 3]:
            self.corr += self.eval_corr(start_seq)

        reward = self.corr

        return (self.logp + reward) / float(self.leng - 1 + 1e-6)


def beam_decode(target_tensor, decoder_hiddens, decoder, input_seq, ignore_mutations=[]):
    """
    :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of
     the output sentence
    :param decoder_hiddens: input tensor of shape [1, B, H] for start of the decoding
    #:param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the
     maximum length of input sentence
    :param decoder:
    :param input_seq
    :param ignore_mutations
    :return: decoded_batch
    """
    counter = count()
    beam_width = 1000
    topk = 1000  # how many sentences do you want to generate
    decoded_batch = []

    # decoding goes sentence by sentence
    for idx in range(1):  # target_tensor.size(0)
        if isinstance(decoder_hiddens, tuple):  # LSTM case
            decoder_hidden = (decoder_hiddens[0][:, idx, :].unsqueeze(0), decoder_hiddens[1][:, idx, :].unsqueeze(0))
        else:
            decoder_hidden = decoder_hiddens[:, idx, :].unsqueeze(0)
        # encoder_output = encoder_outputs[:, idx, :].unsqueeze(1)

        # Start with the start of the sentence token
        decoder_input = torch.LongTensor([[SOS_token]], device=device)

        # Number of sentence to generate
        endnodes = []
        number_required = min((topk + 1), topk - len(endnodes))

        # starting node -  hidden vector, previous node, word id, logp, length
        node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1, [data_lang.index_word[decoder_input.item()]],
                              0)
        nodes = PriorityQueue()

        # start the queue
        nodes.put((-node.eval(input_seq), next(counter), node))
        qsize = 1

        # start beam search
        while True:
            # give up when decoding takes too long
            if qsize > 100000:
                break

            # fetch the best node
            score, _, n = nodes.get()

            decoder_input = n.wordid
            decoder_hidden = n.h

            if (n.wordid.item() == EOS_token or n.wordid.item() == other_token) and n.prevNode is not None:
                endnodes.append((score, next(counter), n))
                # if we reached maximum # of sentences required
                if len(endnodes) >= number_required:
                    break
                else:
                    continue

            # decode for one step using decoder
            decoder_output, decoder_hidden = decoder.forward_step(decoder_input, decoder_hidden)

            # PUT HERE REAL BEAM SEARCH OF TOP
            log_prob, indexes = torch.topk(decoder_output, beam_width)
            print("INDEXES")
            print(models.decode_ids(indexes.squeeze(), data_lang.index_word, end=False))
            quit()
            nextnodes = []

            for new_k in range(beam_width):
                decoded_t = indexes[0][0][new_k].view(1, -1)
                if decoded_t.item() == 0:
                    continue
                decoded_t_id = data_lang.index_word[decoded_t.item()].upper()
                # print(decoded_t, decoded_t_id)
                log_p = log_prob[0][0][new_k].item()

                if decoded_t.item() not in target_tensor and decoded_t_id not in ignore_mutations \
                        and decoded_t_id not in n.seq:

                    node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1,
                                          n.seq + [decoded_t_id], n.corr)

                    score = -node.eval(input_seq)
                    nextnodes.append((score, node))

            # print([(x[0], x[1].seq[-1]) for x in nextnodes])
            # put them into queue
            for i in range(len(nextnodes)):
                score, nn = nextnodes[i]
                nodes.put((score, next(counter), nn))
                # increase qsize
            qsize += len(nextnodes) - 1

        # print([(x[0], x[2].seq[-1]) for x in list(nodes.queue)])
        # choose nbest paths, back trace them
        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]

        utterances = []
        # print([(x[0], x[-1].seq[-1]) for x in endnodes])
        for score, _, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = [n.wordid.squeeze()]
            # back trace
            while n.prevNode is not None:
                n = n.prevNode
                utterance.append(n.wordid.squeeze())

            utterance = utterance[::-1]
            utterances.append(utterance)

        decoded_batch.append(utterances)

    return decoded_batch[0]
