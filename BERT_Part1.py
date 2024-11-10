#bert_env_works
import os
import torch
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


print(f"Torch version: {torch.__version__}")
print(f"Is CUDA available? {torch.cuda.is_available()}")

# Load the full sexism dataset
dataset_path = '/Users/eceserin/Desktop/BAOR/TEZ/TEZ/sexism_data.csv'
if os.path.exists(dataset_path):
    print("File found.")
    sexism_data = pd.read_csv(dataset_path)
    print("Dataset loaded successfully.")
else:
    print("File not found at the specified path.")
    exit()

# Extract the text and label columns
bias_sentences = sexism_data['text'].values.tolist()
bias_labels = sexism_data['sexist'].apply(lambda x: 1 if x == True else 0).values.tolist()

# Initialize tokenizer and BERT model for sequence classification
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
bert_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Tokenize and encode the full dataset
def encode_sentences(sentences):
    return tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, max_length=128)

# Now using the full dataset instead of a smaller subset
encodings = encode_sentences(bias_sentences)
input_ids = encodings['input_ids']
attention_mask = encodings['attention_mask']
labels = torch.tensor(bias_labels)

# Split into train and eval datasets (80/20 split)
train_input_ids, eval_input_ids, train_attention_mask, eval_attention_mask, train_labels, eval_labels = train_test_split(
    input_ids, attention_mask, labels, test_size=0.2, random_state=42
)

# Dataset class
class BiasDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

# Create train and eval datasets
train_dataset = BiasDataset(train_input_ids, train_attention_mask, train_labels)
eval_dataset = BiasDataset(eval_input_ids, eval_attention_mask, eval_labels)

# Define training arguments for full model training
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="steps",  
    save_steps=500,  
    learning_rate=5e-5,
    report_to="none"
)


# Define Trainer with train and eval datasets
trainer = Trainer(
    model=bert_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Train the model with the full dataset
print("Starting training on the full dataset...")
trainer.train()
print("Training completed.")



# Save the fine-tuned model
fine_tuned_model_path = '/Users/eceserin/Desktop/BAOR/TEZ/TEZ/fine_tuned_bert_sexism_full'

# Save the model in PyTorch .bin format
bert_model.save_pretrained(fine_tuned_model_path, safe_serialization=False)  
tokenizer.save_pretrained(fine_tuned_model_path)
trainer.save_model(fine_tuned_model_path)

print(f"Fine-tuned BERT model saved at: {fine_tuned_model_path}")

