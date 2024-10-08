import torch
from queue import PriorityQueue
from itertools import count
import operator
import pickle
import get_reward_values as rew
import random
import pandas as pd
# import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path_database = "/home/scotty/Downloads/code_new/mutation_database/"
hidden_size = 256
batch_size = 1
SOS_token = 2
EOS_token = 3
other_token = 0


def get_reward(curr_node, curr_node_nuc, start_seq, curr_seq):
    reward = 0
    reward_inner = 0
    curr_seq = [x for x in curr_seq if x.upper() not in ["<START>", "<END>"]]

    for mut in curr_seq:
        value = interaction_dict.get(frozenset([get_id(mut), curr_node]), 0)
        #if value == 7:
        #    return 0,0
        reward_inner += value

    if curr_node not in start_seq_scores_dict:
        for mut in start_seq:
            reward += interaction_dict.get(frozenset([get_id(mut), curr_node]), 0)
        reward /= len(start_seq)
        start_seq_scores_dict[curr_node] = reward
    else:
        reward += start_seq_scores_dict[curr_node]

    if len(curr_seq) > 0 and reward_inner == 0 and reward < 3:
        return 0, 0

    return (reward + reward_inner) / (len(curr_seq) + 1), reward_inner


def clean_up_queue(old_queue, full=True):
    sets_list = set()
    new_queue = PriorityQueue()
    cpoint = []
    dups_dict = {}
    length = 0

    while True:
        if old_queue.empty():
            break

        set_add = True
        dups_len = len(sets_list)
        node = old_queue.get()
        length = node[0]

        if node[1] == 0:
            continue

        node_frozenset = frozenset(node[-1].seqid)

        sets_list.add(node_frozenset)
        if len(sets_list) == dups_len:
            continue

        keys = list(dups_dict.keys())

        if full:
            for key in keys:
                intersect = node_frozenset.intersection(key)
                if len(intersect) == len(key) - 1:
                    set_add = False
                    break

        if set_add:
            dups_dict[node_frozenset] = node
            new_queue.put(node)
            cpoint.append(node[-1].seq + [node[1]])

    checkpoint(cpoint, len(cpoint), "j"+str(length)+"_")
    return new_queue


def checkpoint(set_lists, no_, identifier="a"):
    beam_list = ["%s\n" % j for j in set_lists]
    with open("./check_" + identifier + str(no_) + ".txt", "w") as file:
        file.write("".join(beam_list))


with open("./data_lang_full.pkl", "rb") as fo:
    data_lang = pickle.load(fo)

with open("./id_dict_reverse.pkl", "rb") as fo:
    id_dict_reverse = pickle.load(fo)

with open("./interaction_dict_normalized.pkl", "rb") as fo:
    interaction_dict = pickle.load(fo)

with open("./set_dict_05.pkl", "rb") as fo:
    lineages_50p_dict = pickle.load(fo)

start_seq_scores_dict = {}


def get_id(mutation):
    if mutation is None or mutation.lower() in ["<start>", "OOV", "<end>"]:
        return None
    position = int(mutation[1:-1])

    big_dict = id_dict_reverse.get(position, None)
    if big_dict is None:
        return None

    return id_dict_reverse[position].get(mutation, None)


def get_top_indices(index_tensor, beam_length, start_seq, curr_seq, ignorables):
    new_scores = []
    id_dict = {}

    for new_k in range(beam_length):
        decoded_t = index_tensor[0][0][new_k].view(1, -1)
        if decoded_t.item() == 0:
            continue
        decoded_t_nuc = data_lang.index_word[decoded_t.item()].upper()
        if decoded_t_nuc == "OOV":
            print("OOV", decoded_t.item())
            continue
        decoded_t_id = get_id(decoded_t_nuc)

        if decoded_t_nuc not in start_seq + curr_seq + ignorables:
            reward = get_reward(decoded_t_id, decoded_t_nuc, start_seq, curr_seq)
            if reward[0] != 0:
                new_scores.append((reward[0], new_k, decoded_t_nuc))  # FOR CHANGE

    sorted_values = sorted(new_scores, reverse=True)
    probs = torch.softmax(torch.Tensor([x[0] for x in sorted_values]), dim=0)
    cumsums = torch.cumsum(probs, dim=0)
    nucleus = cumsums < 0.9
    filtered_probs = probs[nucleus]
    new_probs = (filtered_probs / sum(filtered_probs)).squeeze().tolist()

    return [x[1] for x in sorted_values[:len(new_probs)]], new_probs, {x[1]: x[0] for x in sorted_values}  # TO CHANGE


class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length, seq, seqID):
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
        self.seqid = seqID
        self.id = get_id(self.seq[-1])
        self.parameters = rew.get_parameters(self.id)
        self.score = 0

    def eval(self, score):
        model_score = (self.logp + self.parameters)  # / float(self.leng - 1 + 1e-6)
        self.score += score
        if self.leng % 5 == 0:
            self.score /= 5

        return model_score, self.score


def beam_decode(target_tensor, decoder_hiddens, decoder, input_seq, ignorables):
    """
    :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of
     the output sentence
    :param decoder_hiddens: input tensor of shape [1, B, H] for start of the decoding
    #:param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the
     maximum length of input sentence
    :param decoder:
    :param input_seq
    :return: decoded_batch
    """
    counter = count()
    beam_width = 100
    beam_full = 1000
    topk = 50000  # how many sentences do you want to generate
    decoded_batch = []
    max_output = 30

    # decoding goes sentence by sentence
    for idx in range(1):  # target_tensor.size(0)
        if isinstance(decoder_hiddens, tuple):  # LSTM case
            decoder_hidden = (decoder_hiddens[0][:, idx, :].unsqueeze(0), decoder_hiddens[1][:, idx, :].unsqueeze(0))
        else:
            decoder_hidden = decoder_hiddens[:, idx, :].unsqueeze(0)
        # encoder_output = encoder_outputs[:, idx, :].unsqueeze(1)

        # Start with the start of the sentence token
        decoder_input = torch.LongTensor([[SOS_token]], device=device)

        # CUSTOM decode for one stop to force the two start tokens
        decoder_output, decoder_hidden = decoder.forward_step(decoder_input, decoder_hidden)
        # _, topi = decoder_output.topk(1)

        # Number of sentence to generate
        endnodes = []
        number_required = min((topk + 1), topk - len(endnodes))

        # starting node -  hidden vector, previous node, word id, logp, length, seq
        node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1, [data_lang.index_word[decoder_input.item()]],
                              [])
        nodes = PriorityQueue()

        # start the queue
        nodes.put((len(node.seq), 0, 0, next(counter), node))
        qsize = 1
        curr_length = 2

        # start beam search
        while True:
            print("qsize", qsize, len(endnodes), curr_length)
            if nodes.empty():
                break

            next_node = nodes.queue[0]

            if len(next_node[-1].seq) < max_output and next_node[0] > curr_length:
                curr_length = next_node[0]
                if curr_length == 3:
                    nodes = clean_up_queue(nodes, full=False)
                else:
                    nodes = clean_up_queue(nodes, full=False)
                qsize = len(list(nodes.queue))
                print("NEW", qsize, nodes.qsize)

            # fetch the best node
            length, summed, score, _, n = nodes.get()

            decoder_input = n.wordid
            decoder_hidden = n.h

            if len(n.seq) >= max_output:
                endnodes.append((summed, score, next(counter), n))  # TO CHANGE
                if len(endnodes) >= number_required:
                    break
                continue

            if (n.wordid.item() == EOS_token or n.wordid.item() == other_token) and (n.prevNode is not None):
                if len(n.seqid) > 5:
                    endnodes.append((score, next(counter), n))
                # if we reached maximum # of sentences required
                if len(endnodes) >= number_required:
                    break
                else:
                    continue
            # decode for one step using decoder
            decoder_output, decoder_hidden = decoder.forward_step(decoder_input, decoder_hidden)

            # PUT HERE REAL BEAM SEARCH OF TOP
            log_prob, indexes = torch.topk(decoder_output, beam_full)
            top_indices, top_probs, top_scores_dict = get_top_indices(indexes, beam_full, input_seq, n.seq, ignorables)

            nextnodes = []
            sample_take = set(random.choices(top_indices, top_probs, k=30))

            for new_k in sample_take:
                decoded_t = indexes[0][0][new_k].view(1, -1)

                decoded_t_id = data_lang.index_word[decoded_t.item()].upper()
                if decoded_t_id == "OOV":
                    continue
                log_p = log_prob[0][0][new_k].item()

                node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1,
                                      n.seq + [decoded_t_id], n.seqid + [decoded_t.item()])

                score, summed = node.eval(-top_scores_dict[new_k])

                # inverses the order because priority queue get ascending !!
                nextnodes.append((summed, -score, node))  # TOO CHANGE

            # print(sorted([(x[0], x[1].seq[-1]) for x in nextnodes]))
            # put them into queue
            for i in range(len(nextnodes)):
                summed, score, nn = nextnodes[i]      # TO OCHANGE
                nodes.put((len(nn.seq), summed, score, next(counter), nn))
                # increase qsize
            qsize += len(nextnodes) - 1

            # print("queue", [(x[1], x[3].seq) for x in list(nodes.queue)[:10]])

        # choose nbest paths, back trace them
        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]

        utterances = []
        # print([(x[0], x[-1].seq[-1]) for x in endnodes])
        for summed, score, _, n in sorted(endnodes, key=operator.itemgetter(0)):  #TO CHANGE
            utterance = [n.wordid.squeeze()]
            # back trace
            while n.prevNode is not None:
                n = n.prevNode
                utterance.append(n.wordid.squeeze())

            utterance = utterance[::-1]
            utterances.append(utterance)

        decoded_batch.append(utterances)

    return decoded_batch[0]
