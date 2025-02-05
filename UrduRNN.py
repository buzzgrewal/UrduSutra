import streamlit as st
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import random
from torch.utils.data import Dataset, DataLoader, random_split

st.title("UrduSutra: The Roman Poetry Weaver")
st.sidebar.title("Model Configuration and Modes")
mode = st.sidebar.radio("Select Mode", ("Train", "Test"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if "training_active" not in st.session_state:
    st.session_state.training_active = False

class PoetryDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length
    def __len__(self):
        return len(self.data) - self.seq_length
    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_length]
        y = self.data[idx + 1 : idx + self.seq_length + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1):
        super(CharRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    def forward(self, x, hidden=None):
        x = self.embed(x)
        output, hidden = self.lstm(x, hidden)
        output = output.contiguous().view(-1, output.shape[2])
        logits = self.fc(output)
        return logits, hidden

def text_to_int(text, char2idx):
    return [char2idx[ch] for ch in text]

def int_to_text(indices, idx2char):
    return ''.join([idx2char[idx] for idx in indices])

def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    loaded_model = CharRNN(checkpoint['vocab_size'],
                           checkpoint['embed_size'],
                           checkpoint['hidden_size'],
                           checkpoint['num_layers']).to(device)
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()
    char2idx = checkpoint['char2idx']
    idx2char = checkpoint['idx2char']
    return loaded_model, char2idx, idx2char

def generate_text(model, start_text, char2idx, idx2char, generation_length=200, temperature=0.8):
    model.eval()
    input_indices = [char2idx.get(ch, 0) for ch in start_text]
    input_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0).to(device)
    hidden = None
    generated_text = start_text
    for _ in range(generation_length):
        logits, hidden = model(input_tensor, hidden)
        logits = logits[-1] / temperature
        probabilities = torch.softmax(logits, dim=0).detach().cpu().numpy()
        next_char_idx = np.random.choice(len(probabilities), p=probabilities)
        next_char = idx2char[next_char_idx]
        generated_text += next_char
        input_tensor = torch.tensor([[next_char_idx]], dtype=torch.long).to(device)
    return generated_text

if mode == "Train":
    st.header("Training Mode")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if "Poetry" not in df.columns:
            st.error("CSV must contain a 'Poetry' column.")
        else:
            poetry_texts = df['Poetry'].dropna().tolist()
            all_text = "\n".join(poetry_texts)
            st.text_area("Data Preview (first 500 characters)", all_text[:500], height=150)
            chars = sorted(list(set(all_text)))
            vocab_size = len(chars)
            char2idx = {ch: idx for idx, ch in enumerate(chars)}
            idx2char = {idx: ch for idx, ch in enumerate(chars)}
            all_data = text_to_int(all_text, char2idx)
            st.sidebar.subheader("Model Configuration")
            SEQ_LENGTH = st.sidebar.number_input("Sequence Length", min_value=10, max_value=500, value=100, step=10)
            BATCH_SIZE = st.sidebar.number_input("Batch Size", min_value=16, max_value=256, value=64, step=16)
            NUM_EPOCHS = st.sidebar.number_input("Number of Epochs", min_value=1, max_value=100, value=5, step=1)
            LEARNING_RATE = st.sidebar.number_input("Learning Rate", min_value=0.0001, max_value=0.01, value=0.003, step=0.0001, format="%.4f")
            EMBED_SIZE = st.sidebar.number_input("Embedding Size", min_value=64, max_value=512, value=128, step=64)
            HIDDEN_SIZE = st.sidebar.number_input("Hidden Size", min_value=128, max_value=1024, value=256, step=128)
            NUM_LAYERS = st.sidebar.number_input("Number of Layers", min_value=1, max_value=4, value=2, step=1)
            full_dataset = PoetryDataset(all_data, SEQ_LENGTH)
            dataset_size = len(full_dataset)
            train_size = int(0.9 * dataset_size)
            test_size = dataset_size - train_size
            train_dataset, _ = random_split(full_dataset, [train_size, test_size])
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
            model = CharRNN(vocab_size, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            criterion = nn.CrossEntropyLoss()

            if not st.session_state.training_active:
                if st.button("Start Training"):
                    st.session_state.training_active = True
            else:
                if st.sidebar.button("Stop Training"):
                    st.session_state.training_active = False

            progress_bar = st.progress(0)
            progress_text = st.empty()
            epoch_details = st.empty()
            # Use a text_area with fixed height for batch details
            batch_details = st.empty()
            total_epochs = int(NUM_EPOCHS)
            epoch_info_str = ""
            if st.session_state.training_active:
                for epoch in range(1, total_epochs + 1):
                    if not st.session_state.training_active:
                        epoch_details.info("Training stopped by user.")
                        break
                    epoch_loss = 0.0
                    batch_info_str = ""
                    for batch_idx, (inputs, targets) in enumerate(train_loader):
                        if not st.session_state.training_active:
                            batch_details.text_area("Batch Details", "Training stopped by user during batch processing.", height=150)
                            break
                        inputs, targets = inputs.to(device), targets.to(device)
                        optimizer.zero_grad()
                        logits, _ = model(inputs)
                        loss = criterion(logits, targets.view(-1))
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                        batch_info_str += f"Epoch {epoch} Batch {batch_idx+1} Loss: {loss.item():.4f}\n"
                        batch_details.text_area("Batch Details", batch_info_str, height=150)
                    avg_loss = epoch_loss / len(train_loader)
                    epoch_info_str += f"Epoch {epoch} Average Loss: {avg_loss:.4f}\n"
                    epoch_details.text(epoch_info_str)
                    percentage = int((epoch / total_epochs) * 100)
                    progress_bar.progress(percentage)
                    progress_text.text(f"Training Progress: {percentage}%")
                st.session_state.training_active = False
                model_path = "char_rnn_model.pth"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'vocab_size': vocab_size,
                    'embed_size': EMBED_SIZE,
                    'hidden_size': HIDDEN_SIZE,
                    'num_layers': NUM_LAYERS,
                    'char2idx': char2idx,
                    'idx2char': idx2char,
                }, model_path)
                st.success(f"Model saved to {model_path}")

if mode == "Test":
    st.header("Testing Mode")
    model_file = st.file_uploader("Upload Model File", type=["pth"])
    if model_file is not None:
        with open("temp_model.pth", "wb") as f:
            f.write(model_file.getbuffer())
        loaded_model, loaded_char2idx, loaded_idx2char = load_model("temp_model.pth", device)
        prompt = st.text_input("Enter a prompt", value="pyar")
        generation_length = st.slider("Generation Length", min_value=50, max_value=500, value=200, step=10)
        temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=0.8, step=0.1)
        if st.button("Generate"):
            generated_poetry = generate_text(loaded_model, prompt, loaded_char2idx, loaded_idx2char, generation_length, temperature)
            st.text_area("Generated Poetry", generated_poetry, height=300)
