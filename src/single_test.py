import data_handler2 as dh
import pickle
import data_loader as dl
import models
import random
import torch
# to import files from parent directories, we use package relative imports
# we have to have a __init__.py file in each, to be recognized as packages
# then if we execute the script itll show "atempted relative import with
# no known parent package", because it thinks that it is at the top
# of the module hierarchy
import beam_decode as beam
import collections


def get_input_ref(seq, data_processor):
    seq_encoded, _ = dh.tokenize([" ".join(seq)], data_processor)
    return seq_encoded


def preprocess_seq(seq, data_processor, dev):
    inpt_tensor, _ = dh.tokenize([dh.preprocess_sentences(seq)], data_processor)
    return torch.LongTensor(inpt_tensor).to(dev)


# fix variables
def create_beams(seq_tensor, input_ref_seq_encoded, input_ref_seq, encoder, decoder, data_processor):
    encoder_outputs, encoder_hidden = encoder(seq_tensor)
    decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden)

    beams = beam.beam_decode(input_ref_seq_encoded[0], encoder_hidden, decoder, input_ref_seq)
    full_list = []
    for bm in beams:
        bm_decoded = models.decode_ids(bm, data_processor.index_word)
        # print(bm_decoded)

        full_list.append([x.upper() for x in bm_decoded] + input_ref_seq)

    # print("MOSTCOMMON", collections.Counter(full_list).most_common()[:50])

    # return [x[0].upper() for x in collections.Counter(full_list).most_common()[:50]]
    return full_list


def encode_ids(muts, vocab):
    encoded = []
    for mt in muts:
        encoded.append(vocab[mt])

    return encoded


def beamify_list(list_sequences, analysis_ref, data_processor, encoder, decoder, dev):
    evolved_seqs = []
    for seq in list_sequences:
        seq_full = seq + analysis_ref
        seq_full.sort(key=lambda x: int(x[1:-1]))

        seq_encoded = get_input_ref(seq_full, data_processor)
        seq_tensor = preprocess_seq(seq_full, data_processor, dev)

        beams = create_beams(seq_tensor, seq_encoded, seq_full, encoder, decoder, data_processor)
        # remove beams
        beams = [x for x in beams if x not in to_ignore]
        #print("all", beams)

        # sampling method
        beams = [random.choice(beams)]
        print("chosen", beams)

        evolved_seqs += [seq + [predicted_mut] for predicted_mut in beams]

    return evolved_seqs


# INITIAL DATA
path = "C:\\Users\\zhasmi00\\Downloads\\test\\ba286_test.txt"
fasta_path = "C:\\Users\\zhasmi00\\Downloads\\test\\"
fasta_name = "beams"

with open(path, "r") as file:
    input_muts = file.readline().replace(" ", "").split(",")

with open("data/data_lang_full.pkl", "rb") as fo:
    data_lang = pickle.load(fo)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_size = 256
batch_size = 1
ref_seq = [x for x in dl.REFSEQ]
mutated_ref_seq = dh.get_full_seq(ref_seq, input_muts)

max_output = 36
sos_token = data_lang.word_index["<start>"]
vocab_length = len(data_lang.word_index)

# INITIALIZE ENCODER AND DECODER
enc = models.EncoderRNN(vocab_length+1, hidden_size).to(device)
dec = models.DecoderRNN(hidden_size, vocab_length+1, max_output, sos_token).to(device)

enc.load_state_dict(torch.load("./models2/enc/enc.pth", map_location=torch.device("cpu")))
dec.load_state_dict(torch.load("./models2/dec/dec.pth", map_location=torch.device("cpu")))
enc.eval()
dec.eval()

# MUTATIONS TO IGNORE
to_ignore = []


# GENERATE BEGINNING
input_muts.sort(key=lambda x: int(x[1:-1]))

input_seq_encoded = get_input_ref(input_muts, data_lang)
input_tensor = preprocess_seq(input_muts, data_lang, device)

output_muts = create_beams(input_tensor, input_seq_encoded, input_muts, enc, dec, data_lang)

dh.create_fasta("resFINAL", "./", [dh.get_full_seq(ref_seq, x) for x in output_muts],
                                   dh.get_full_seq(ref_seq, input_muts))
