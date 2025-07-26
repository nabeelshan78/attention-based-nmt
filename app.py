import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, AdditiveAttention, Concatenate
from tensorflow.keras.models import Model
import numpy as np
import json
import os # For checking file existence
# For downloading large files from Hugging Face Hub (if applicable for deployment)
# from huggingface_hub import hf_hub_download 

# --- Model Definition and Loading (as provided by you) ---

# Load the vocabulary files
@st.cache_data # Cache vocabulary loading
def load_vocab(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Update path to dataset/processed as per your latest code
eng_vocab_path = 'dataset/processed/eng_vocab.json'
fra_vocab_path = 'dataset/processed/fra_vocab.json'

# Ensure vocab files exist for the app to run
if not os.path.exists(eng_vocab_path) or not os.path.exists(fra_vocab_path):
    st.error(f"Error: Vocabulary files '{eng_vocab_path}' and/or '{fra_vocab_path}' not found.")
    st.stop() # Stop the Streamlit app if files are missing

eng_vocab = load_vocab(eng_vocab_path)
fra_vocab = load_vocab(fra_vocab_path)

eng_inv_vocab = {idx: word for word, idx in eng_vocab.items()}
fra_inv_vocab = {idx: word for word, idx in fra_vocab.items()}

input_vocab_size = len(eng_vocab)
target_vocab_size = len(fra_vocab)
embedding_dim = 128
Tx = 22
Ty = 20
lstm_units = 128

@st.cache_resource # Cache the model creation and weight loading
def get_nmt_models(Tx, Ty, input_vocab_size, target_vocab_size, embedding_dim, lstm_units):
    """
    Builds a Sequence-to-Sequence NMT model with Bidirectional LSTM encoder,
    Additive Attention, and Unidirectional LSTM decoder. Returns models for
    training and inference.
    """
    embedding_dim_for_attention = lstm_units 

    # --- Shared Layers (to reuse weights between training and inference models) ---
    encoder_embedding_layer = Embedding(input_vocab_size, embedding_dim_for_attention, mask_zero=True, name="encoder_embedding")
    encoder_bilstm = Bidirectional(LSTM(lstm_units, return_sequences=True, return_state=True, dropout=0.3), name="bi_encoder_lstm")
    encoder_outputs_projection_layer = Dense(lstm_units, activation="tanh", name="encoder_outputs_projection")

    decoder_embedding_layer = Embedding(target_vocab_size, embedding_dim_for_attention, mask_zero=True, name="decoder_embedding") 
    attention_layer = AdditiveAttention(name="attention_layer")
    decoder_lstm = LSTM(lstm_units * 2, return_sequences=True, return_state=True, dropout=0.3, name="decoder_lstm")
    output_dense_layer = Dense(target_vocab_size, activation='softmax', name="output_dense")

    # --- 1. Training Model Architecture ---
    encoder_inputs = Input(shape=(Tx,), name="encoder_inputs")
    enc_embedding_output = encoder_embedding_layer(encoder_inputs)
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_bilstm(enc_embedding_output)
    encoder_outputs_projected = encoder_outputs_projection_layer(encoder_outputs)
    state_h = Concatenate(name="decoder_initial_h")([forward_h, backward_h])
    state_c = Concatenate(name="decoder_initial_c")([forward_c, backward_c])

    decoder_inputs = Input(shape=(Ty,), name="decoder_inputs") 
    dec_embedding_output = decoder_embedding_layer(decoder_inputs)
    context_vector = attention_layer([dec_embedding_output, encoder_outputs_projected])
    decoder_combined_input = Concatenate(axis=-1, name="decoder_combined_input")([dec_embedding_output, context_vector])
    decoder_outputs, _, _ = decoder_lstm(decoder_combined_input, initial_state=[state_h, state_c])
    decoder_predictions = output_dense_layer(decoder_outputs)
    training_model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_predictions, name="nmt_training_model")

    # --- 2. Encoder Inference Model ---
    encoder_inference_model = Model(inputs=encoder_inputs,
                                    outputs=[encoder_outputs_projected, state_h, state_c],
                                    name="encoder_inference_model")

    # --- 3. Decoder Inference Model ---
    decoder_input_single_token = Input(shape=(1,), name="decoder_input_single_token")
    encoder_states_input = Input(shape=(Tx, lstm_units,), name="encoder_states_input")
    decoder_state_h_input = Input(shape=(lstm_units * 2,), name="decoder_h_input")
    decoder_state_c_input = Input(shape=(lstm_units * 2,), name="decoder_c_input")

    dec_single_embedding_output = decoder_embedding_layer(decoder_input_single_token) 
    inference_context_vector = attention_layer([dec_single_embedding_output, encoder_states_input])
    inference_decoder_combined_input = Concatenate(axis=-1, name="decoder_combined_input_inference")([dec_single_embedding_output, inference_context_vector])

    inference_decoder_outputs, h_state_output, c_state_output = decoder_lstm(
        inference_decoder_combined_input,
        initial_state=[decoder_state_h_input, decoder_state_c_input]
    )
    inference_predictions = output_dense_layer(inference_decoder_outputs)

    decoder_inference_model = Model(
        inputs=[decoder_input_single_token, encoder_states_input, decoder_state_h_input, decoder_state_c_input],
        outputs=[inference_predictions, h_state_output, c_state_output],
        name="decoder_inference_model"
    )

    # --- Model Weights Loading Logic (can be updated for Hugging Face Hub if needed) ---
    model_weights_path = 'model_weights.h5'
    # If using Hugging Face Hub for weights, uncomment and modify these lines:
    # MODEL_REPO_ID = "your-username/your-nmt-model-repo" 
    # MODEL_FILENAME = "model_weights.h5"
    # try:
    #     model_weights_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME)
    #     st.success(f"Model weights downloaded from Hugging Face Hub: {model_weights_path}")
    # except Exception as e:
    #     st.error(f"Error downloading model weights from Hugging Face Hub: {e}. Please ensure '{MODEL_FILENAME}' exists in your repo '{MODEL_REPO_ID}' and is accessible.")
    #     st.warning("The application will continue with randomly initialized weights.")
    #     model_weights_path = None # Indicate that weights were not loaded

    if os.path.exists(model_weights_path):
        try:
            training_model.load_weights(model_weights_path)
            st.success("Model loaded successfully.")
        except Exception as e:
            st.error(f"Error loading weights: {e}")
            st.warning("The application will continue with randomly initialized weights.")
    else:
        st.warning(f"Model weights file '{model_weights_path}' not found. The application will use randomly initialized weights.")
    
    return training_model, encoder_inference_model, decoder_inference_model, \
           output_dense_layer, attention_layer, decoder_lstm, decoder_embedding_layer

model, encoder_inference_model, decoder_inference_model, \
    output_dense_layer, attention_layer, decoder_lstm, decoder_embedding_layer = \
    get_nmt_models(Tx, Ty, input_vocab_size, target_vocab_size, embedding_dim, lstm_units)


# --- Utility Functions (as provided by you) ---
def tokenize(sentences):
    """
    Tokenizes a list of sentences by lowercasing and splitting by space.
    """
    tokenized = []
    for sentence in sentences:
        tokens = sentence.lower().strip().split()
        tokenized.append(tokens)
    return tokenized

def sentences_to_indices(tokenized_sentences, vocab):
    """
    Converts tokenized sentences into sequences of vocabulary indices.
    Uses '<unk>' token for out-of-vocabulary words.
    """
    return [
        [vocab.get(word, vocab["<unk>"]) for word in sentence]
        for sentence in tokenized_sentences
    ]

# The predict_translation function is crucial and needs to be cached for performance.
@st.cache_data(show_spinner=False) # show_spinner=False because we'll add our own spinner
def predict_translation_cached(input_sentence, encoder_inference_model, decoder_inference_model,
                               input_vocab, input_inv_vocab, target_vocab, target_inv_vocab,
                               Tx, Ty, beam_width=3, max_decoder_steps=None):
    """
    Translates a single English sentence to French using the trained NMT model
    with beam search. This version is wrapped for Streamlit caching.
    """
    if max_decoder_steps is None:
        max_decoder_steps = Ty 

    tokenized_input = tokenize([input_sentence])
    indexed_input = sentences_to_indices(tokenized_input, input_vocab)
    
    padded_input = tf.keras.preprocessing.sequence.pad_sequences(
        indexed_input, maxlen=Tx, padding='post', value=input_vocab["<pad>"]
    )
    padded_input = np.array(padded_input) 

    encoder_outputs, h, c = encoder_inference_model.predict(padded_input, verbose=0)

    beams = [([target_vocab["<sos>"]], h, c, 0.0)]
    final_translations = [] 

    for _ in range(max_decoder_steps):
        all_candidates = []
        for seq, h_state, c_state, log_prob in beams:
            if seq[-1] == target_vocab["<eos>"]:
                final_translations.append((seq, log_prob))
                continue 

            last_token = seq[-1]
            decoder_single_input = np.array([[last_token]]) 

            decoder_outputs_probs, new_h, new_c = decoder_inference_model.predict(
                [decoder_single_input, encoder_outputs, h_state, c_state], verbose=0
            )

            token_probs = decoder_outputs_probs[0, 0, :] 
            top_k_indices = np.argsort(token_probs + 1e-10)[-beam_width:][::-1] 

            for token_idx in top_k_indices:
                token_log_prob = np.log(token_probs[token_idx] + 1e-10) 
                candidate_seq = seq + [token_idx] 
                candidate_log_prob = log_prob + token_log_prob 
                all_candidates.append((candidate_seq, new_h, new_c, candidate_log_prob))
        
        all_candidates.sort(key=lambda x: x[3], reverse=True)
        beams = all_candidates[:beam_width]

        if all(b_seq[-1] == target_vocab["<eos>"] for b_seq, _, _, _ in beams):
            final_translations.extend([(b_seq, b_log_prob) for b_seq, _, _, b_log_prob in beams if b_seq[-1] == target_vocab["<eos>"]])
            break 

    if not final_translations and not beams:
        return "Translation failed: No valid sequences generated."
    
    if not final_translations:
        best_seq, _, _, best_log_prob = max(beams, key=lambda x: x[3])
    else: 
        best_seq, best_log_prob = max(final_translations, key=lambda x: x[1])

    translated_words = []
    for idx in best_seq:
        if idx == target_vocab["<sos>"] or idx == target_vocab["<pad>"]:
            continue
        if idx == target_vocab["<eos>"]:
            break 
        translated_words.append(target_inv_vocab.get(idx, '<unk>')) 

    return " ".join(translated_words)


# --- Streamlit UI ---

# Set wide mode and light theme
st.set_page_config(layout="wide", page_title="NMT with Attention", initial_sidebar_state="expanded")

# Custom CSS for light theme and general aesthetics
st.markdown(
    """
    <style>
    /* General body and app background */
    body {
        color: #333; /* Dark text for light background */
        background-color: #f0f2f6; /* Light grey background */
    }
    .stApp {
        background-color: #f0f2f6;
    }
    /* Card-like containers for sections */
    /* Removed .stButton from here as it caused issues with padding for the button itself */
    .stExpander, .stTextInput, .stTextArea, .stSelectbox, .stRadio, .stSlider { 
        background-color: #ffffff; /* White background for UI elements */
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08); /* Subtle shadow */
        margin-bottom: 20px; /* Space between sections */
        border: 1px solid #e0e0e0; /* Light border */
    }
    /* Input fields */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: #f8f8f8; /* Slightly off-white for input fields */
        color: #333;
        border-radius: 8px;
        border: 1px solid #d0d0d0;
        padding: 10px;
    }
    /* Selectbox dropdown arrow */
    .stSelectbox>div>div>div>div:last-child {
        background-color: #f8f8f8;
        border-radius: 0 8px 8px 0;
    }
    /* Buttons */
    .stButton>button {
        background-color: #4CAF50; /* Green button */
        color: white;
        border-radius: 8px;
        border: none;
        padding: 12px 24px; /* Slightly larger padding */
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 17px; /* Slightly larger font */
        margin: 4px 2px;
        cursor: pointer;
        transition: all 0.3s ease; /* Smooth transition for hover */
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); /* Subtle shadow for buttons */
        width: 100%; /* Make button span full width of its container */
    }
    .stButton>button:hover {
        background-color: #45a049; /* Darker green on hover */
        transform: translateY(-2px); /* Slight lift effect */
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15); /* More prominent shadow on hover */
    }
    /* Headings */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #1a1a1a; /* Darker headings */
        font-weight: 600; /* Slightly bolder */
    }
    /* Alerts */
    .stAlert {
        border-radius: 8px;
        margin-top: 15px;
        margin-bottom: 15px;
        padding: 15px;
        font-size: 15px;
    }
    .stAlert.st-success {
        background-color: #e6ffe6; /* Light green for success */
        color: #1f7a1f;
        border: 1px solid #a3e6a3;
    }
    .stAlert.st-error {
        background-color: #ffe6e6; /* Light red for error */
        color: #cc0000;
        border: 1px solid #ffb3b3;
    }
    .stAlert.st-warning {
        background-color: #fff8e6; /* Light yellow for warning */
        color: #e69100;
        border: 1px solid #ffd9b3;
    }
    .stAlert.st-info {
        background-color: #e6f7ff; /* Light blue for info */
        color: #007acc;
        border: 1px solid #b3e0ff;
    }

    /* Overall page padding */
    .reportview-container .main .block-container {
        padding-top: 3rem; /* More top padding */
        padding-right: 4rem; /* More horizontal padding */
        padding-left: 4rem;
        padding-bottom: 3rem;
    }

    /* Styling for the radio buttons (input method selection) */
    div.stRadio > label {
        background-color: #f8f8f8; /* Light background for radio button labels */
        border: 1px solid #d0d0d0;
        border-radius: 8px;
        padding: 10px 15px;
        margin-right: 10px;
        cursor: pointer;
        transition: background-color 0.2s ease, border-color 0.2s ease;
    }
    div.stRadio > label:hover {
        background-color: #e8e8e8; /* Slightly darker on hover */
        border-color: #b0b0b0;
    }
    div.stRadio > label > div[data-baseweb="radio"] { /* Specific styling for the radio circle itself */
        border-color: #4CAF50 !important; /* Green border */
    }
    div.stRadio > label > div[data-baseweb="radio"] > div { /* Fill color when selected */
        background-color: #4CAF50 !important;
    }

    /* Specific styling for the output container */
    .output-container {
        background-color: #f8f8f8; /* Light grey background for output */
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        border: 1px solid #e0e0e0;
        margin-top: 30px; /* Space above output */
        margin-bottom: 20px;
    }
    .output-container h4 {
        color: #1a1a1a;
        margin-bottom: 10px;
    }
    .output-container p {
        font-size: 1.1em;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- App Title and Introduction ---
st.title("🧠 Neural Machine Translation (NMT) with Attention")

st.markdown("""
<div style='text-align: center; font-size: 1.1em; margin-bottom: 30px;'>
    Welcome to this interactive demonstration of a cutting-edge Sequence-to-Sequence Neural Machine Translation model, powered by an **attention mechanism**!
    This application allows you to translate English sentences to French, showcasing the power of deep learning in language processing.
</div>
""", unsafe_allow_html=True)


# --- Model Explanation Section (Collapsible) ---
with st.expander("✨ Understand How It Works", expanded=True):
    st.subheader("How Our Translation Brain Works:")
    st.markdown(
        """
        Imagine our translator as a team of specialized AI agents working together to understand and speak French. Here's what each part does:

        -   **The Listener (Encoder - Bidirectional LSTM):** This is the part that *listers* carefully to your entire English sentence. It reads each word, not just from left to right, but also from right to left (that's the "Bidirectional" magic!). This helps it build a super rich understanding of every word's meaning in its context, creating a detailed **"thought summary"** of the whole sentence.

        -   **The Focuser (Attention Mechanism - Additive Attention):** As our translator starts speaking French, this agent continuously asks, "Which part of the original English sentence should I *focus* on right now to say this next French word?" It's like your eyes scanning a text, highlighting the most important bits. This **dynamic focusing** is key, especially for long sentences, because it ensures the translation stays accurate and relevant, preventing words from getting lost or misunderstood.

        -   **The Speaker (Decoder - Unidirectional LSTM):** This agent takes the "thought summary" from the Listener and, guided by the Focuser, starts *speaking* the French translation one word at a time. It uses what it's already said and what the Focuser highlights to decide the next best word, making sure the French sentence flows naturally.

        -   **The Smart Guesser (Beam Search):** Instead of just picking the single "most likely" next word (which can sometimes lead to mistakes), our translator uses a clever strategy called Beam Search. It considers **several top possible translation paths** at each step, like exploring a few different routes on a map. By keeping track of the best multiple options, it can ultimately choose the most coherent and accurate full French sentence, significantly improving translation quality.
        """
    )
    st.markdown("---") # Added a subtle separator within the expander


# Sample sentences
english_sentences = [
    "You asked all the right questions.",
    "How are you?", "I love machine learning.", "What time is it?", 
    "Where is the nearest station?", "Thank you very much!", 
    "I want to go home.", "The weather is beautiful today.", 
    "Can I help you?", "Please speak slowly.", "I don't understand.",
    "What is your name?", "My name is John.", "I am a student.",
    "He likes to read books.", "She plays the piano.", "We are learning French.",
    "They live in Paris.", "This is a difficult problem.", "I need some water.",
    "Do you speak English?", "Yes, a little.", "No, not at all.",
    "Good morning.", "Good night.", "See you later.", "Excuse me.",
    "I am sorry.", "It's a beautiful day.", "The cat is on the table.",
    "Where is the bathroom?", "I would like a coffee, please.",
    "How much does it cost?", "I am hungry.", "I am thirsty.",
    "Help me, please.", "Call an ambulance!", "I am lost.",
    "Can you show me the way?", "I live in Lahore.", "My favorite color is blue.",
    "This food is delicious.", "I enjoy traveling.", "What is your job?",
    "I am an engineer.", "It's raining outside.", "Happy birthday!",
    "I wish you good luck.", "The world is big.", "Tell me more.",
    "I need a break."
]

# --- Main Translation Interface ---
st.subheader("🚀 Translate Your English Sentence to French!")

# Input and Beam Width in columns
col1, col2 = st.columns([3, 1]) 

with col1:
    translation_option = st.radio(
        "**1. Choose an input method:**",
        ("Select from samples", "Enter your own English sentence"),
        horizontal=True,
        key="input_method_radio"
    )

    input_sentence = ""

    if translation_option == "Select from samples":
        input_sentence = st.selectbox(
            "Select an English sentence to translate:",
            ["-- Select a sentence --"] + english_sentences,
            index=0,
            key="sample_sentence_select"
        )
        if input_sentence == "-- Select a sentence --":
            input_sentence = ""
    else:
        input_sentence = st.text_area(
            "Enter your English sentence here:",
            placeholder="e.g., I am learning about neural networks.",
            height=100,
            key="custom_text_input"
        ).strip()

with col2:
    st.markdown("<br>", unsafe_allow_html=True) # Add some vertical space
    beam_width = st.slider(
        "**2. Adjust Beam Width:**",
        min_value=1, max_value=10, value=3, step=1,
        help="Higher beam width explores more translation paths, potentially yielding better results but takes longer.",
        key="beam_width_slider"
    )

# --- Translate Button and Output Section ---
# This section is now outside the columns to span the full width
st.markdown("<br>", unsafe_allow_html=True) # Add some vertical space before the button

# The single, native Streamlit button
if st.button("✨ Translate!", key="translate_button_main", use_container_width=True):
    if input_sentence:
        # Placeholder for output to update dynamically
        # This placeholder needs to be defined outside the button's if block
        # but its content updated inside.
        # For simplicity and direct display, we can just render the output directly
        # after the button click, which Streamlit handles well.
        with st.spinner("Translating... This might take a moment due to model inference."):
            try:
                translated_text = predict_translation_cached(
                    input_sentence,
                    encoder_inference_model,
                    decoder_inference_model,
                    eng_vocab,
                    eng_inv_vocab,
                    fra_vocab,
                    fra_inv_vocab,
                    Tx,
                    Ty,
                    beam_width=beam_width
                )
                # Display output in a styled container directly
                st.markdown(f"""
                    <div class="output-container">
                        <h4>Translation Result:</h4>
                        <p><b>Original English:</b> {input_sentence}</p>
                        <p><b>Translated French:</b> <span style="color: #007acc; font-weight: bold;">{translated_text}</span></p>
                    </div>
                """, unsafe_allow_html=True)
                st.success("Translation Complete!") # Keep the success alert
            except Exception as e:
                st.error(f"An error occurred during translation: {e}")
                st.warning("Please ensure your `dataset/processed` directory with `eng_vocab.json` and `fra_vocab.json` exists, and `model_weights.h5` is in the same directory as this script, or configured for Hugging Face Hub download.")
    else:
        st.warning("Please select a sentence or enter text to translate before clicking 'Translate!'.")

st.markdown("---")
st.markdown("""
<div style='text-align: center; margin-top: 30px; font-size: 0.9em; color: #666;'>
    **Developed by an aspiring AI/ML Engineer with a passion for cutting-edge research, aiming for impactful publications and a career at a leading tech company.**
</div>
""", unsafe_allow_html=True)