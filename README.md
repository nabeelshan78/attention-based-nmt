# 🌐 Neural Machine Translation (NMT) Project

---

## ✨ Project Overview

This repository showcases a robust Neural Machine Translation (NMT) system, meticulously developed from data preprocessing to model training and inference. The project implements an Attention-based Bidirectional LSTM Sequence-to-Sequence model, a powerful architecture capable of translating text between languages.

This project serves as a strong demonstration of my capabilities in deep learning, natural language processing, and building end-to-end AI solutions.

---

## 🚀 Key Features & Highlights

- **End-to-End NMT Pipeline**: Covers the entire workflow from raw text data to trained translation model.

- **Advanced Sequence-to-Sequence Architecture**:
  - **Bidirectional LSTM Encoder**: Captures context from both directions of the input sequence.
  - **Additive Attention Mechanism**: Dynamically focuses on relevant parts of the source sentence during decoding, significantly improving translation quality.
  - **LSTM Decoder**: Generates the target language sequence.

- **Efficient Data Preprocessing**: Custom scripts for cleaning, tokenization, vocabulary building, and numericalization of text data.

- **TensorFlow/Keras Implementation**: Leverages TensorFlow 2.x and Keras for efficient model building, training, and inference.

- **Optimized Training**: Incorporates advanced training techniques such as tf.data pipelines for efficient data loading, ModelCheckpoint for saving best models, EarlyStopping for preventing overfitting, and ReduceLROnPlateau for adaptive learning rate scheduling.

- **Beam Search Decoding**: Implements beam search for improved translation quality during inference, exploring multiple translation hypotheses.

- **Version Control for Large Files (Git LFS)**: Utilizes Git LFS to manage large preprocessed .npy datasets and model checkpoints, keeping the repository lightweight and efficient.

---

## 🧠 Model Architecture

The core of this NMT system is a Sequence-to-Sequence (Seq2Seq) model with an Attention mechanism:

- **Encoder**: A Bidirectional LSTM network processes the input (English) sentence, generating a rich contextual representation for each word and a final hidden state.

- **Decoder**: An LSTM network generates the output (French) sentence one word at a time. At each decoding step, an Additive Attention layer calculates a context vector by weighting the encoder's outputs based on the current decoder state, allowing the decoder to "look back" at relevant parts of the source sentence.

- **Shared Embeddings & Output Layer**: Embedding layers and the final dense output layer are shared between training and inference models for consistent weight usage.

---

## 📂 Project Structure
```bash
attention-based-nmt/
├── checkpoints/                        # Ignored by Git LFS: Stores trained model checkpoints (.keras files)
├── dataset/
│   ├── processed/                      # Processed data and vocabularies
│   │   ├── decoder_input_data.npy      # Decoder input sequences (LFS tracked)
│   │   ├── decoder_target_data.npy     # Decoder target sequences (LFS tracked)
│   │   ├── encoder_input_data.npy      # Encoder input sequences (LFS tracked)
│   │   ├── eng_fra_clean_filtered.txt  # Cleaned sentence pairs
│   │   ├── eng_vocab.json              # English vocabulary (word to index)
│   │   ├── fra_vocab.json              # French vocabulary (word to index)
│   │   ├── test.txt                    # Test split
│   │   ├── train.txt                   # Training split
│   │   └── val.txt                     # Validation split
│   └── raw/                            # Raw dataset files
│       ├── cleaned_eng_fra.txt         # Intermediate cleaned raw data
│       └── eng_fra.txt                 # Original raw English-French dataset
├── .gitattributes                      # Git LFS configuration for tracking large files
├── .gitignore                          # Specifies files/directories to ignore (e.g., checkpoints, venv)
├── clean_data.ipynb                    # Jupyter Notebook for initial data cleaning
├── model_training.ipynb                # Main Jupyter Notebook for model definition, training, and evaluation
├── preprocessing.py                    # Python script containing data preprocessing functions
├── processing.ipynb                    # Jupyter Notebook for further data processing (tokenization, vocab building)
├── translated_comparison.tsv           # Output TSV for translation comparison (generated)
├── translation_comparison.tsv          # Another output TSV for translation comparison (generated)
└── translations_output.tsv             # Output TSV for bulk translations (generated)
```
---


---

## ⚙️ Getting Started

To set up and run this project locally:

### Clone the repository:

```bash
git clone https://github.com/nabeelshan78/attention-based-nmt.git
cd attention-based-nmt
Open model_training.ipynb to execute the full pipeline.
```
---

## Training Progress & Results
The model was trained for multiple epochs, demonstrating consistent improvement in validation loss and accuracy. Early stopping and learning rate reduction strategies were employed to optimize training.

## Training & Validation Loss/Accuracy Plot
![Training and Validation History](path/to/your/training_history.png)


## Sample Translations
Here are a few examples of translations generated by the model during inference, showcasing its capabilities and areas for potential improvement:
## 🧪 Sample Translations

Here are a few examples of translations generated by the model during inference. These samples demonstrate both the model's capabilities and areas where it may require further refinement:

| **Original English**           | **Model's French Translation**         | **Back-Translated English**       |
|-------------------------------|----------------------------------------|-----------------------------------|
| How are you?                  | comment vas-tu ?                       | how are you ?                     |
| I love machine learning.      | j'adore les mathématiques.             | i love mathematics                |
| What time is it?              | quelle heure est-il, ?                 | what time is it, ?                |
| Where is the nearest station? | où se trouve la gare la plus proche ? | where is the nearest station ?    |
| Thank you very much!          | merci beaucoup !                       | thank you very much !             |
| I want to go home.            | je veux aller à la maison.             | i want to go home                 |
| Call an ambulance!            | appelle un bisou !                     | call a kiss !                     |
| I live in Lahore.             | je vis confortablement.                | i live comfortably                |

> ⚠️ **Note:** Some translations may exhibit minor inaccuracies or "hallucinations"  
> (e.g., *"j'adore les mathématiques"* for *"I love machine learning"*).  
> These are common challenges in Neural Machine Translation (NMT) and highlight opportunities  
> for improved training data, fine-tuning, or model architecture enhancements.
---

## 🛠️ Skills Demonstrated

This project effectively showcases my proficiency in the following areas:

- **Deep Learning & Neural Networks**  
  Designing, implementing, and training complex Sequence-to-Sequence (Seq2Seq) models.

- **Natural Language Processing (NLP)**  
  Text preprocessing, tokenization, vocabulary creation, and sequence modeling.

- **TensorFlow & Keras**  
  Advanced use of `tf.keras` APIs for model construction, training pipelines with `tf.data`, and custom layer creation.

- **Attention Mechanisms**  
  Integration of attention layers for enhanced sequence alignment and translation accuracy.

- **Model Optimization Techniques**  
  Use of callbacks such as `EarlyStopping`, `ReduceLROnPlateau`, and `ModelCheckpoint` for better training control.

- **Inference Techniques**  
  Implementation of **beam search** decoding for generating high-quality translation outputs.

- **Data Handling**  
  Efficient handling of large datasets using NumPy and JSON-based pipelines.

- **Python Programming**  
  Writing modular, readable, and efficient Python code following best practices.

- **Version Control**  
  Managing codebase and large assets with **Git** and **Git LFS**.

- **Problem Solving & Debugging**  
  Iterative model improvement through rigorous debugging and evaluation.

---

## 🚀 Future Enhancements

Potential next steps to improve this project:

- **📊 Larger Dataset & Pre-trained Embeddings**  
  Incorporate larger corpora and pre-trained embeddings such as **GloVe** or **FastText** to boost performance.

- **🧠 Transformer Architecture**  
  Transition from Seq2Seq with attention to a full **Transformer-based** architecture for state-of-the-art results.

- **⚙️ Multi-GPU / Distributed Training**  
  Implement multi-GPU support using TensorFlow’s `MirroredStrategy` for faster training.

- **📉 Model Compression**  
  Explore **quantization**, **pruning**, or **knowledge distillation** for efficient deployment.

- **📏 Evaluation Metrics**  
  Add **BLEU score** and other NMT-specific metrics for quantitative evaluation of translation quality.

- **🌐 Interactive Web Demo**  
  Build a live demo interface using **Streamlit**, **Gradio**, or similar tools for real-time translation testing.
