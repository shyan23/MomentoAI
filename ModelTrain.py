from supabase import Client,create_client
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import re
import string
import emoji
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import words
import contractions
nltk.download('words')
from spellchecker import SpellChecker
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download("punkt_tab")
from datasets import load_dataset
ds = load_dataset("kartikay/review-summarizer")

df = ds['train'].to_pandas()

from sklearn.model_selection import train_test_split


df_train, df_temp = train_test_split(df, train_size=50000, random_state=42)
df_validation, df_test_temp = train_test_split(df_temp, train_size=20000, random_state=42)
df_test, _ = train_test_split(df_test_temp, train_size=20000, random_state=42)


from transformers import AutoTokenizer

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-xsum-9-6")
max_length = 512  # Define your max sequence length

def tokenize_in_batches(df, batch_size=1000):
    # Tokenize in smaller batches
    tokenized_data = {"input_ids": [], "attention_mask": [], "labels": []}
    for start in range(0, len(df), batch_size):
        end = start + batch_size
        batch = df.iloc[start:end]

        # Tokenize input text and target in batch
        inputs = tokenizer(
            list(batch["text"]),
            max_length=max_length,
            truncation=True,
            padding="max_length"
        )
        labels = tokenizer(
            list(batch["target"]),
            max_length=max_length,
            truncation=True,
            padding="max_length"
        )

        # Add to tokenized data
        tokenized_data["input_ids"].extend(inputs["input_ids"])
        tokenized_data["attention_mask"].extend(inputs["attention_mask"])
        tokenized_data["labels"].extend(labels["input_ids"])

    return tokenized_data

tokenized_train = tokenize_in_batches(df_train)
tokenized_validation = tokenize_in_batches(df_validation)
tokenized_test = tokenize_in_batches(df_test)


from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# Load model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-xsum-9-6")
tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-xsum-9-6")

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)
model = get_peft_model(model, lora_config)

# Train as usual, memory usage will be much lower!


from torch.utils.data import DataLoader, TensorDataset
import torch

# Convert tokenized data into PyTorch tensors
train_dataset = TensorDataset(
    torch.tensor(tokenized_train["input_ids"]),
    torch.tensor(tokenized_train["attention_mask"]),
    torch.tensor(tokenized_train["labels"]),
)

validation_dataset = TensorDataset(
    torch.tensor(tokenized_validation["input_ids"]),
    torch.tensor(tokenized_validation["attention_mask"]),
    torch.tensor(tokenized_validation["labels"]),
)

# Create DataLoaders
batch_size = 2  # Adjust based on your GPU memory
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)

from torch.optim import AdamW
from transformers import get_scheduler

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Scheduler
num_training_steps = len(train_dataloader) * 3  # 3 epochs
lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)



from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from torch.nn import CrossEntropyLoss
import torch
from tqdm import tqdm  # For progress bar
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()  # For scaling gradients during mixed precision

# Move model to GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
epochs = 3  # Number of epochs
gradient_accumulation_steps = 4  # Accumulate gradients for larger effective batch size

for epoch in range(epochs):
    model.train()
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
    for batch in progress_bar:
        # Move batch to GPU
        input_ids, attention_mask, labels = (b.to(device) for b in batch)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss / gradient_accumulation_steps  # Normalize loss

        # Backward pass
        loss.backward()

        # Update weights
        if (batch[0].shape[0] % gradient_accumulation_steps == 0):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        progress_bar.set_postfix(loss=loss.item())

    # Validation after each epoch
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in validation_dataloader:
            input_ids, attention_mask, labels = (b.to(device) for b in batch)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            val_loss += outputs.loss.item()

    print(f"Validation Loss after epoch {epoch + 1}: {val_loss / len(validation_dataloader)}")


