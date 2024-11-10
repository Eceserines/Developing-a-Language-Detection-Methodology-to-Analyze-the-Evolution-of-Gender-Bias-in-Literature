#USE ENV_CNN_CLEAN

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Dropout
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import re

# Load the Word2Vec model
word2vec_model_path = '/Users/eceserin/Desktop/BAOR/TEZ/TEZ/word2vec_sexism.bin'
word2vec_model = KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True)

print("Word2Vec model loaded.")

# Load the corresponding labels and sentences from the dataset
dataset_path = '/Users/eceserin/Desktop/BAOR/TEZ/TEZ/sexism_data.csv'
sexism_data = pd.read_csv(dataset_path)
y = sexism_data['sexist'].apply(lambda x: 1 if x == True else 0).values 
sentences = sexism_data['text'].values  # Store the sentences

# Define a function to get the average Word2Vec embeddings for each sentence
def get_sentence_embedding(sentence, word2vec_model, embedding_dim=100):
    tokens = re.findall(r'\b\w+\b', sentence.lower()) 
    valid_tokens = [token for token in tokens if token in word2vec_model]  
    if not valid_tokens:
        return np.zeros(embedding_dim)  # Return a zero vector if no valid tokens
    embedding = np.mean([word2vec_model[token] for token in valid_tokens], axis=0)
    return embedding

# Create sentence embeddings for all the sentences in the dataset
embedding_dim = 100  
X = np.array([get_sentence_embedding(sentence, word2vec_model, embedding_dim) for sentence in sentences])


# Split into train and test sets
X_train, X_test, y_train, y_test, sentences_train, sentences_test = train_test_split(X_scaled, y, sentences, test_size=0.2, random_state=42)

# Reshape the embeddings for Conv1D input (samples, timesteps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build an enhanced CNN model for binary classification
def build_enhanced_cnn_model(learning_rate):
    model = Sequential()
    
    # First Conv1D layer with Batch Normalization
    model.add(Conv1D(128, 5, activation='relu', input_shape=(X_train.shape[1], 1), padding='same'))
    model.add(BatchNormalization())  
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    
    # Second Conv1D layer with Batch Normalization
    model.add(Conv1D(64, 5, activation='relu', padding='same')) 
    model.add(BatchNormalization()) 
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    
    # GlobalMaxPooling and fully connected layers
    model.add(GlobalMaxPooling1D())
    
    # Fully connected (Dense) layer with ReLU activation
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))  
    model.add(Dropout(0.4))
    
    # Final output layer using sigmoid for binary classification
    model.add(Dense(1, activation='sigmoid'))  
    
    # Compile the model with Adam optimizer and binary_crossentropy loss for classification
    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Train the enhanced CNN model for more epochs
learning_rate = 0.0001
model = build_enhanced_cnn_model(learning_rate)
history = model.fit(X_train, y_train, epochs=4, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy}')

# Predict on the test set (sequentially)
y_pred_test = (model.predict(X_test).flatten() > 0.5).astype(int)  

# SSave the predicted scores and sentences to an Excel file
data_all = {'Tested Sentence': sentences_test, 'True Label': y_test, 'Predicted Label': y_pred_test}
df_all = pd.DataFrame(data_all)
output_path_all = '/Users/eceserin/Desktop/BAOR/TEZ/TEZ/cnn_test_results_with_word2vec_sentences.xlsx'
df_all.to_excel(output_path_all, index=False)
print(f"Test results (sentences, true labels, and predicted labels) saved to {output_path_all}")


'''


# Calculate and print the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_test)
print("Confusion Matrix:")
print(conf_matrix)

# Plot the confusion matrix using matplotlib
plt.figure(figsize=(6, 4))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix for CNN with Word2Vec')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Non-Sexist', 'Sexist'], rotation=45)
plt.yticks(tick_marks, ['Non-Sexist', 'Sexist'])

# Print numbers inside the confusion matrix
thresh = conf_matrix.max() / 2.
for i, j in np.ndindex(conf_matrix.shape):
    plt.text(j, i, format(conf_matrix[i, j], 'd'),
             horizontalalignment="center",
             color="white" if conf_matrix[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred_test, target_names=['Non-Sexist', 'Sexist']))

# Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_test)
plt.plot(recall, precision, marker='.')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()

# ROC Curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_pred_test)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='r')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


# Plot training and validation accuracy over epochs (adjusted to start epochs from 1)
plt.figure(figsize=(8, 6))
plt.plot(np.arange(1, len(history.history['acc'])+1), history.history['acc'], label='Train Accuracy')  # Adjust x-axis
plt.plot(np.arange(1, len(history.history['val_acc'])+1), history.history['val_acc'], label='Validation Accuracy')  # Adjust x-axis
plt.title('Model Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()'''


#TEST ON THE BOOKS
'''
# Test on the textbook data
#textbook_path = '/Users/eceserin/Desktop/BAOR/TEZ/TEZ/processed_sentences_His Family.txt'
#textbook_path = '/Users/eceserin/Desktop/BAOR/TEZ/TEZ/processed_sentences_Trust.txt'
textbook_path = '/Users/eceserin/Desktop/BAOR/TEZ/TEZ/Sentences from Agent 123.txt'
with open(textbook_path, 'r', encoding='utf-8') as file:
    textbook_data = file.readlines()


textbook_sentences = [sentence.strip() for sentence in textbook_data if sentence.strip()]
print(f"Number of textbook sentences: {len(textbook_sentences)}")



# Predict on the textbook data
y_pred_textbook = (model.predict(X_textbook_scaled).flatten() > 0.5).astype(int)

# Save the predictions for textbook data
data_textbook = {'Tested Sentence': textbook_sentences, 'Predicted Label': y_pred_textbook}
df_textbook = pd.DataFrame(data_textbook)
output_path_textbook = '/Users/eceserin/Desktop/BAOR/TEZ/TEZ/cnn_textbook_results_with_word2vec_sentences_agent123.xlsx'
df_textbook.to_excel(output_path_textbook, index=False)
print(f"Test results for textbook data saved to {output_path_textbook}")

# Calculate the percentage of non-sexist sentences in the textbook
non_sexist_count = np.sum(y_pred_textbook == 0)
total_sentences = len(y_pred_textbook)
non_sexist_percentage = (non_sexist_count / total_sentences) * 100
print(f"Percentage of Non-Sexist Sentences in the textbook: {non_sexist_percentage:.2f}%")


    
 
# Define the list of target words
target_words = ["man", "men", "woman", "women", "he", "she", "girl", "boy", "girls", "boys"]

# Predict on the textbook data and get raw prediction scores (weights)
y_pred_textbook_scores = model.predict(X_textbook_scaled).flatten()


# Print the top 5 sentences with highest and lowest prediction scores
# Top 5 highest prediction scores (most sexist predictions)
top_5_highest = np.argsort(y_pred_textbook_scores)[-100:][::-1]
print("\nTop 5 Most Sexist Predictions (Highest Scores):")
for idx in top_5_highest:
    print(f"Sentence: {textbook_sentences[idx]}\nScore: {y_pred_textbook_scores[idx]}\n")
    


# Top 5 lowest prediction scores (most non-sexist predictions)
top_5_lowest = np.argsort(y_pred_textbook_scores)[:15]
print("\nTop 5 Least Sexist Predictions (Lowest Scores):")
for idx in top_5_lowest:
    print(f"Sentence: {textbook_sentences[idx]}\nScore: {y_pred_textbook_scores[idx]}\n")

# Filter and print the top 5 least sexist sentences that contain any target word
filtered_least_sexist = pd.DataFrame({
    'Sentence': textbook_sentences,
    'Score': y_pred_textbook_scores
}).sort_values(by='Score').reset_index(drop=True)  # Sort by score (ascending)

# Filter for non-sexist predictions and those that contain any target word
filtered_least_sexist = filtered_least_sexist[filtered_least_sexist['Score'] <= 0.5]
filtered_least_sexist['Contains Target Word'] = filtered_least_sexist['Sentence'].apply(
    lambda sentence: any(word in sentence.lower().split() for word in target_words)
)
filtered_least_sexist_with_target = filtered_least_sexist[filtered_least_sexist['Contains Target Word']]

# Get the top 5 least sexist sentences with target words
top_5_least_sexist_with_target = filtered_least_sexist_with_target.nsmallest(15, 'Score')

# Print the results
print("\nTop 5 Least Sexist Sentences Containing Target Words:")
for idx, row in top_5_least_sexist_with_target.iterrows():
    print(f"Sentence: {row['Sentence']}\nScore: {row['Score']}\n")
    
    
    

# Plot the distribution of prediction scores (weights) as a histogram
plt.figure(figsize=(10, 6))
n, bins, patches = plt.hist(y_pred_textbook_scores, bins=20, color='gray', edgecolor='black')

# Set color based on the bin center
for i, patch in enumerate(patches):
    if bins[i] >= 0.5:  # Set red color for bins with center >= 0.5
        patch.set_facecolor('red')
    else:  # Set green color for bins with center < 0.5
        patch.set_facecolor('green')

# Add labels and title
plt.xlabel('Prediction Score (Weight)')
plt.ylabel('Count')  # Changed from 'Density' to 'Count'
plt.title('Distribution of Prediction Scores in Trust')
plt.xticks(np.arange(0, 1.1, 0.1))  # Set x-axis ticks at intervals of 0.1

# Show the plot
plt.show()
'''


#TEST ON THE AGENTS'S FINDINGS

import numpy as np
import pandas as pd

# Load and clean the textbook data
#textbook_path = '/Users/eceserin/Desktop/BAOR/TEZ/TEZ/Sentences from Agent 123.txt'
textbook_path = '/Users/eceserin/Desktop/BAOR/TEZ/TEZ/Where does agent 1 break.txt'
with open(textbook_path, 'r', encoding='utf-8') as file:
    textbook_data = file.readlines()

# Prepare the textbook sentences by stripping whitespace
textbook_sentences = [sentence.strip() for sentence in textbook_data if sentence.strip()]
print(f"Number of textbook sentences: {len(textbook_sentences)}")

# Create embeddings for textbook sentences
X_textbook = np.array([get_sentence_embedding(sentence, word2vec_model, embedding_dim) for sentence in textbook_sentences])


# Predict on the textbook data to get both labels and prediction scores (weights)
y_pred_textbook_scores = model.predict(X_textbook_scaled).flatten()
y_pred_textbook_labels = (y_pred_textbook_scores > 0.5).astype(int)  # Binary labels based on threshold

# Save the predictions for textbook data with both sentences and scores (weights) in an Excel file
data_textbook = {
    'Tested Sentence': textbook_sentences,
    'Prediction Score (Weight)': y_pred_textbook_scores
}
df_textbook = pd.DataFrame(data_textbook)
output_path_textbook = '/Users/eceserin/Desktop/BAOR/TEZ/TEZ/where does agent 2 break.xlsx'
df_textbook.to_excel(output_path_textbook, index=False)
print(f"Test results for textbook data saved to {output_path_textbook}")

