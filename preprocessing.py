import tensorflow as tf
import numpy as np



def filter_and_count_sentences(input_path, max_words=20):
    """
    Reads a parallel corpus file, filters sentence pairs where either the English
    or French sentence exceeds max_words, and returns statistics.

    Args:
        input_path (str): Path to the dataset file (tab-separated English ↔ French).
        max_words (int): Maximum allowed words in both English and French sentences.

    Returns:
        tuple:
            - list: Filtered lines (kept sentence pairs as raw lines).
            - int: Total number of sentence pairs read.
            - int: Number of pairs exceeding max_words in either sentence.
            - int: Number of sentences exceeding max_words in English.
            - int: Number of sentences exceeding max_words in French.
    """
    total_sentences = 0
    filtered_sentences = []
    exceeding_pairs = 0
    english_exceeding = 0
    french_exceeding = 0

    with open(input_path, "r", encoding="utf-8") as infile:
        for line in infile:
            total_sentences += 1
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue  # Skip lines that don't have both English and French

            eng = parts[0].strip()
            fra = parts[1].strip()

            eng_words = eng.split()
            fra_words = fra.split()

            eng_len = len(eng_words)
            fra_len = len(fra_words)

            eng_too_long = eng_len > max_words
            fra_too_long = fra_len > max_words

            if eng_too_long:
                english_exceeding += 1
            if fra_too_long:
                french_exceeding += 1

            if eng_too_long or fra_too_long:
                exceeding_pairs += 1
            else:
                filtered_sentences.append(line.strip())

    return filtered_sentences, total_sentences, exceeding_pairs, english_exceeding, french_exceeding




def remove_attributes_from_filtered_sentences(filtered_sentences_with_attributes):
    """
    Takes a list of raw lines from the filtered dataset and removes the
    attribution information, returning only the English and French sentences.

    Args:
        filtered_sentences_with_attributes (list): A list of strings,
                                                   each containing "Eng\tFra\tAttr".

    Returns:
        list: A list of strings, each containing "Eng\tFra".
    """
    clean_sentences = []
    for line in filtered_sentences_with_attributes:
        # Split by the tab character. The first two parts are English and French.
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            # Reconstruct the line with only English and French, joining with a tab
            clean_sentences.append(f"{parts[0]}\t{parts[1]}")
        else:
            # Handle malformed lines if they exist
            print(f"Warning: Skipping malformed line: {line.strip()}")
    return clean_sentences



# Load sentence pairs
def load_cleaned_pairs(clean_sentences):
    pairs = [line.split('\t') for line in clean_sentences]
    return pairs


def tokenize(sentences):
    tokenized = []
    for i, sentence in enumerate(sentences):
        tokens = sentence.lower().strip().split()
        tokenized.append(tokens)
    return tokenized



# Build vocabulary dictionary and inverse vocabulary
def build_vocab(tokenized_sentences):
    vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
    index = 4
    for sentence in tokenized_sentences:
        for word in sentence:
            if word not in vocab:
                vocab[word] = index
                index += 1

    # Create inverse vocabulary
    inv_vocab = {idx: word for word, idx in vocab.items()}

    return vocab, inv_vocab



def sentences_to_indices(tokenized_sentences, vocab):
    return [
        [vocab.get(word, vocab["<unk>"]) for word in sentence]
        for sentence in tokenized_sentences
    ]



def preprocess_data(pairs, input_vocab, target_vocab, Tx, Ty):
    """
    Preprocesses the data for the NMT model: tokenizes, numericalizes,
    adds SOS/EOS tokens, and pads sequences.

    Args:
        pairs (list): A list of [English_sentence, French_sentence] pairs.
        input_vocab (dict): Vocabulary for the input (English) language.
        target_vocab (dict): Vocabulary for the target (French) language.
        Tx (int): Maximum sequence length for the encoder input (English).
        Ty (int): Maximum sequence length for the decoder target (French, including SOS/EOS).

    Returns:
        tuple:
            - np.array: Encoder input data (padded English sequences).
            - np.array: Decoder input data (padded French sequences, shifted).
            - np.array: Decoder target data (padded French sequences, actual labels).
    """
    # Separate English and French sentences
    eng_sentences = [pair[0] for pair in pairs]
    fra_sentences = [pair[1] for pair in pairs]

    # Tokenize sentences
    tokenized_eng = tokenize(eng_sentences)
    tokenized_fra = tokenize(fra_sentences)
    

    # Numericalize sentences
    indices_eng = sentences_to_indices(tokenized_eng, input_vocab)
    indices_fra = sentences_to_indices(tokenized_fra, target_vocab)

    # Add SOS and EOS tokens to French sentences for decoder input/output
    # Decoder input will start with <sos> and end before <eos>
    # Decoder target will start after <sos> and include <eos>
    decoder_input_indices = []
    decoder_target_indices = []

    for fra_seq in indices_fra:
        # Decoder input: <sos> + French sentence (without <eos>)
        decoder_input_indices.append([target_vocab["<sos>"]] + fra_seq)
        # Decoder target: French sentence (without <sos>) + <eos>
        decoder_target_indices.append(fra_seq + [target_vocab["<eos>"]])

    # Pad sequences
    # Encoder input: pad English sequences to Tx
    encoder_input_padded = tf.keras.preprocessing.sequence.pad_sequences(
        indices_eng,
        maxlen=Tx,
        padding='post',
        value=input_vocab["<pad>"]
    )

    # Decoder input: pad French sequences (with <sos>) to Ty-1 (as per model input shape)
    # The decoder input in the model is `Ty-1` because it doesn't predict the final EOS token
    # as part of its *input sequence*, but rather the *output* sequence.
    # We need to ensure the padded length matches the model's expected input length for decoder.
    decoder_input_padded = tf.keras.preprocessing.sequence.pad_sequences(
        decoder_input_indices,
        maxlen=Ty, # Model's decoder_inputs shape is (Ty - 1,)
        padding='post',
        value=target_vocab["<pad>"]
    )

    # Decoder target: pad French sequences (with <eos>) to Ty
    decoder_target_padded = tf.keras.preprocessing.sequence.pad_sequences(
        decoder_target_indices,
        maxlen=Ty,
        padding='post',
        value=target_vocab["<pad>"]
    )

    # Ensure all outputs are numpy arrays
    return np.array(encoder_input_padded), np.array(decoder_input_padded), np.array(decoder_target_padded)
  