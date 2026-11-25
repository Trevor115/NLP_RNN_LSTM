"""
main.py

RNN vs LSTM - Next Word Prediction
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

from data import TEXT


# -----------------------------
# SETTINGS
# -----------------------------
SEQ_LENGTH = 3
EMBED_DIM = 10
HIDDEN_UNITS = 32
EPOCHS = 150
VERBOSE = 0


# -----------------------------
# DATA PREPARATION
# -----------------------------
def prepare_data(text, seq_length=3):
    """
    Convert raw text into training sequences.

    Example:
    input words:  the dog ran
    target word:  across
    """
    # Create a tokenizer
    tokenizer = Tokenizer()

    # Fit the tokenizer on the text
    tokenizer.fit_on_texts([text])

    # Compute the vocabulary size
    total_words = len(tokenizer.word_index) + 1

    # Convert text into encoded integers
    encoded = tokenizer.texts_to_sequences([text])[0]

    X = []
    y = []

    # Build input/output sequences
    # Each X should contain seq_length words
    # Each y should contain the next word after that sequence
    for i in range(seq_length, len(encoded)):
        X.append(encoded[i - seq_length:i])
        y.append(encoded[i])

    # Convert X and y into NumPy arrays
    X = np.array(X)
    y = np.array(y)

    # One-hot encode y
    y = to_categorical(y, num_classes=total_words)

    return tokenizer, total_words, X, y


# -----------------------------
# MODEL BUILDERS
# -----------------------------
def build_rnn_model(vocab_size, seq_length, embed_dim=10, hidden_units=32):
    """
    Build a SimpleRNN next-word prediction model.
    """
    model = Sequential([
        # Add embedding layer
        Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=seq_length),

        # Add SimpleRNN layer
        SimpleRNN(hidden_units),

        # Add Dense output layer with softmax
        Dense(vocab_size, activation="softmax")
    ])

    # Compile the model
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def build_lstm_model(vocab_size, seq_length, embed_dim=10, hidden_units=32):
    """
    Build an LSTM next-word prediction model.
    """
    model = Sequential([
        # Add embedding layer
        Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=seq_length),

        # Replace the SimpleRNN with an LSTM layer
        LSTM(hidden_units),

        # Add Dense output layer with softmax
        Dense(vocab_size, activation="softmax")
    ])

    # Compile the model
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


# -----------------------------
# PREDICTION HELPERS
# -----------------------------
def predict_next_word(model, tokenizer, seed_text, seq_length=3):
    """
    Predict the next word from a prompt.
    """
    # Convert the seed text into token ids
    encoded = tokenizer.texts_to_sequences([seed_text])[0]

    # For this starter version, the prompt must contain exactly seq_length words
    if len(encoded) != seq_length:
        return "[ERROR: prompt must contain exactly 3 words]"

    # Reshape for the model
    encoded = np.array(encoded).reshape(1, seq_length)

    # Predict probabilities
    predicted_probs = model.predict(encoded, verbose=0)

    # Get the index of the highest-probability word
    predicted_index = np.argmax(predicted_probs, axis=1)[0]

    # Convert the predicted index back into a word
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word

    return "[UNKNOWN]"


def test_model_predictions(model, tokenizer, prompts, seq_length=3):
    """
    Run a list of prompts through the model.
    """
    results = []

    for prompt in prompts:
        # Predict the next word for each prompt
        next_word = predict_next_word(model, tokenizer, prompt, seq_length)
        results.append((prompt, next_word))

    return results


def print_results(title, results):
    """
    Print results in a readable way.
    """
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    for prompt, prediction in results:
        print(f"Prompt: {prompt:<20} -> Prediction: {prediction}")


# -----------------------------
# MAIN
# -----------------------------
def main():
    print("Preparing data...")
    tokenizer, vocab_size, X, y = prepare_data(TEXT, seq_length=SEQ_LENGTH)

    print(f"Vocabulary size: {vocab_size}")
    print(f"Training examples: {len(X)}")

    print("\nBuilding models...")
    rnn_model = build_rnn_model(vocab_size, SEQ_LENGTH, EMBED_DIM, HIDDEN_UNITS)
    lstm_model = build_lstm_model(vocab_size, SEQ_LENGTH, EMBED_DIM, HIDDEN_UNITS)

    print("\nTraining SimpleRNN model...")
    # Train the RNN model
    rnn_model.fit(X, y, epochs=EPOCHS, verbose=VERBOSE)

    print("Training LSTM model...")
    # Train the LSTM model
    lstm_model.fit(X, y, epochs=EPOCHS, verbose=VERBOSE)

    # Evaluate both models
    rnn_loss, rnn_acc = rnn_model.evaluate(X, y, verbose=0)
    lstm_loss, lstm_acc = lstm_model.evaluate(X, y, verbose=0)

    print("\nTraining Performance")
    print("-" * 60)
    print(f"SimpleRNN Accuracy: {rnn_acc:.4f}")
    print(f"LSTM Accuracy     : {lstm_acc:.4f}")

    # Prompts must contain exactly 3 words in this starter version
    prompts = [
        "the dog ran",
        "the cat sat",
        "the dog chased",
        "the boy threw",
        "the girl ran",
        "the bird flew"
    ]

    # Test both models
    rnn_results = test_model_predictions(rnn_model, tokenizer, prompts, SEQ_LENGTH)
    lstm_results = test_model_predictions(lstm_model, tokenizer, prompts, SEQ_LENGTH)

    # Print results
    print_results("SimpleRNN Predictions", rnn_results)
    print_results("LSTM Predictions", lstm_results)

    print("\nRequired reflection questions:")
    print("1. Which model had better training accuracy?")
    print("2. Which model gave more sensible next-word predictions?")
    print("3. Where did the SimpleRNN fail?")
    print("4. Why might the LSTM perform better on sequence tasks?")


if __name__ == "__main__":
    main()