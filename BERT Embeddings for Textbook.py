#use bert_env_works
#this one takes around 10 mins


import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load the textbook data
#textbook_path = '/Users/eceserin/Desktop/BAOR/TEZ/TEZ/processed_sentences_Trust.txt'
#textbook_path = '/Users/eceserin/Desktop/BAOR/TEZ/TEZ/Sentences from Agent 1-2.txt'
textbook_path = '/Users/eceserin/Desktop/BAOR/TEZ/TEZ/Where does agent 1 break.txt'
with open(textbook_path, 'r') as file:
    textbook_sentences = [line.strip() for line in file if line.strip()]  # Remove blank lines

print(f"Number of sentences: {len(textbook_sentences)}")

# Load the tokenizer and fine-tuned model
fine_tuned_model_path = '/Users/eceserin/Desktop/BAOR/TEZ/TEZ/fine_tuned_bert_sexism_full'
tokenizer = DistilBertTokenizer.from_pretrained(fine_tuned_model_path)
model_bert = DistilBertForSequenceClassification.from_pretrained(fine_tuned_model_path, output_hidden_states=True)

# Process in batches to prevent memory issues
batch_size = 256 
all_embeddings = []
embedding_dim = 768  
max_tokens = 128  

for i in range(0, len(textbook_sentences), batch_size):
    batch_sentences = textbook_sentences[i:i + batch_size]
    encoded_batch = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors='pt', max_length=max_tokens)

    with torch.no_grad():
        outputs = model_bert(**encoded_batch)
        batch_embeddings = outputs.hidden_states[-1].detach().cpu().numpy()  # Move to CPU if on GPU
    
    # Ensure each batch has a consistent shape
    if batch_embeddings.shape[1] != max_tokens:  
        padded_batch = np.zeros((batch_embeddings.shape[0], max_tokens, embedding_dim))
        if batch_embeddings.shape[1] > max_tokens:
            # Truncate to max_tokens
            padded_batch = batch_embeddings[:, :max_tokens, :]
        else:
            # Pad to max_tokens
            padded_batch[:, :batch_embeddings.shape[1], :] = batch_embeddings
        batch_embeddings = padded_batch
    
    all_embeddings.append(batch_embeddings)

# Concatenate all embeddings into a single numpy array
textbook_embeddings_np = np.concatenate(all_embeddings, axis=0)

# Check the number of embeddings against the number of sentences
print(f"Number of sentences: {len(textbook_sentences)}")
print(f"Number of embeddings: {textbook_embeddings_np.shape[0]}")

# Save embeddings as .npy file
#embedding_save_path = '/Users/eceserin/Desktop/BAOR/TEZ/TEZ/textbook_bert_embeddings_Agents12.npy'
embedding_save_path = '/Users/eceserin/Desktop/BAOR/TEZ/TEZ/where does agent 3 fail.npy'

np.save(embedding_save_path, textbook_embeddings_np)
print(f"BERT embeddings saved to {embedding_save_path}")
