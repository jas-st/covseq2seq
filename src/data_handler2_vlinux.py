import tensorflow as tf


def preprocess_sentences(w: list):
    w = ["<start>"] + w + ["<end>"]
    return w


def tokenize(lang, lang_tokenizer=None):
    # lang = list of sentences in a language [" ".join(x) for x in input_full
    # ex = random.randint(0, len(lang))

    # print(len(lang), "example sentence: {}".format(lang[ex]))
    if lang_tokenizer is None:
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='<OOV>')
        lang_tokenizer.fit_on_texts(lang)

    # tf.keras.preprocessing.text.Tokenizer.texts_to_sequences converts string (w1, w2, w3, ......, wn)
    # to a list of correspoding integer ids of words (id_w1, id_w2, id_w3, ...., id_wn)
    tensor = lang_tokenizer.texts_to_sequences(lang)

    # tf.keras.preprocessing.sequence.pad_sequences takes argument a list of integer id sequences
    # and pads the sequences to match the longest sequences in the given input
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

    return tensor, lang_tokenizer


def get_full_seq(ref, mutations):
    full_seq = [x for x in ref]
    for mut in mutations:
        if mut not in ["<START>", "<start>", "<END>", "<end>", "OOV"]:
            full_seq[int(mut[1:-1]) - 1] = mut[-1]

    return "".join(full_seq)


def create_fasta(file_name, file_path, input_strings, beginning_seq):
    with open(file_path + file_name + ".fasta", 'w') as file:
        if beginning_seq is not None:
            fastastring = ">input\n%s\n" % beginning_seq
            file.write(fastastring)

        fasta_list = [">%s\n%s\n" % (f"beam{j+1}", input_strings[j]) for j in range(len(input_strings))]
        file.write("".join(fasta_list))

