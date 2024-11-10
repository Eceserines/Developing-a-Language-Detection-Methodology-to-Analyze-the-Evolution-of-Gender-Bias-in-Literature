# USE BERT_ENV39

import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import pandas as pd


# Path to the fine-tuned model
fine_tuned_model_path = '/Users/eceserin/Desktop/BAOR/TEZ/TEZ/fine_tuned_bert_sexism_full'

# Load the tokenizer and model using DistilBert
tokenizer = DistilBertTokenizer.from_pretrained(fine_tuned_model_path)
model = DistilBertForSequenceClassification.from_pretrained(fine_tuned_model_path, output_hidden_states=True)


model.eval()

# Define a function to get prediction probabilities and embeddings for full sentences
def get_prediction_weights_and_embeddings(sentence, max_length=165):
    inputs = tokenizer(sentence, return_tensors='pt', max_length=max_length, truncation=True, padding='max_length')
    with torch.no_grad():  # Disable gradient calculation
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)  
        
        
        if hasattr(outputs, 'hidden_states'):
            embeddings = outputs.hidden_states[-1] 
        else:
            embeddings = logits 
    return probabilities[0][1].item(), embeddings.squeeze().numpy()

# Load the sentences from the dataset
dataset_path = '/Users/eceserin/Desktop/BAOR/TEZ/TEZ/sexism_data.csv'
sexism_data = pd.read_csv(dataset_path)
bias_sentences = sexism_data['text'].values.tolist() 

# Generate prediction-based weights and embeddings for all sentences
weights = []
embeddings = []
for sentence in bias_sentences:
    weight, embedding = get_prediction_weights_and_embeddings(sentence)
    weights.append(weight)
    embeddings.append(embedding)

# Convert weights and embeddings to numpy arrays
weights = np.array(weights)
embeddings = np.array(embeddings)

# Save the prediction weights to a NumPy file
weights_path = '/Users/eceserin/Desktop/BAOR/TEZ/TEZ/prediction_weights.npy'
np.save(weights_path, weights)
print(f"Saved prediction-based weights to {weights_path}")

# Save the BERT embeddings to a NumPy file
embeddings_path = '/Users/eceserin/Desktop/BAOR/TEZ/TEZ/bert_supervised_embeddings.npy'
np.save(embeddings_path, embeddings)
print(f"Saved BERT embeddings to {embeddings_path}")
