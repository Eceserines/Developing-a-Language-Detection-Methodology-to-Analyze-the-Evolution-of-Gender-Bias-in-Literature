#USE ENV_CNN_CLEAN


import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Dropout, BatchNormalization
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt





# Load the BERT embeddings generated from the sexism dataset
bert_embeddings_path = '/Users/eceserin/Desktop/BAOR/TEZ/TEZ/bert_supervised_embeddings.npy'
X = np.load(bert_embeddings_path)
print("Embeddings loaded:", X.shape)

# Load the corresponding labels and sentences from the dataset
dataset_path = '/Users/eceserin/Desktop/BAOR/TEZ/TEZ/sexism_data.csv'
sexism_data = pd.read_csv(dataset_path)
y = sexism_data['sexist'].apply(lambda x: 1 if x == True else 0).values  
sentences = sexism_data['text'].values  # Store the sentences

# print the lengths here to confirm sizes before proceeding
print(f"Number of embeddings: {len(X)}")
print(f"Number of labels: {len(y)}")
print(f"Number of sentences: {len(sentences)}")



#  Manually reshape the embeddings to 2D for scaling
X_reshaped = X.reshape((X.shape[0], -1))  
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reshaped)  

# Reshape embeddings back to 3D for Conv1D input
n_samples = X_scaled.shape[0]
X_scaled = X_scaled.reshape((n_samples, 165, -1))  

# Split into train and test sets
X_train, X_test, y_train, y_test, sentences_train, sentences_test = train_test_split(X_scaled, y, sentences, test_size=0.2, random_state=42)

# Build an enhanced CNN model for binary classification
def build_enhanced_cnn_model(learning_rate):
    model = Sequential()
    model.add(Conv1D(128, 5, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    
    model.add(Conv1D(64, 5, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    
    model.add(GlobalMaxPooling1D())
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    
    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

print('CNN model built')

# Train the enhanced CNN model
learning_rate = 0.0001
model = build_enhanced_cnn_model(learning_rate)
history = model.fit(X_train, y_train, epochs=2, batch_size=32, validation_data=(X_test, y_test), verbose=1)
print('CNN model trained')

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy}')

# Predict on the test set
y_pred_test = (model.predict(X_test).flatten() > 0.5).astype(int)

# Save the predicted scores and sentences to an Excel file
data_all = {'Tested Sentence': sentences_test, 'True Label': y_test, 'Predicted Label': y_pred_test}
df_all = pd.DataFrame(data_all)
output_path_all = '/Users/eceserin/Desktop/BAOR/TEZ/TEZ/cnn_test_results_with_sentences.xlsx'
df_all.to_excel(output_path_all, index=False)
print(f"Test results saved to {output_path_all}")



'''

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(6, 4))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix for CNN with BERT Embeddings')
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

# Classification report
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

# Plot training and validation accuracy over epochs
plt.figure(figsize=(8, 6))
plt.plot(np.arange(1, len(history.history['acc'])+1), history.history['acc'], label='Train Accuracy')  # Adjust x-axis
plt.plot(np.arange(1, len(history.history['val_acc'])+1), history.history['val_acc'], label='Validation Accuracy')  # Adjust x-axis
plt.title('Model Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
'''





'''

#TEST
from sklearn.preprocessing import MinMaxScaler

# Load BERT embeddings and perform padding/truncation
#textbook_embeddings_path = '/Users/eceserin/Desktop/BAOR/TEZ/TEZ/textbook_bert_embeddings_His Family.npy'
textbook_embeddings_path = '/Users/eceserin/Desktop/BAOR/TEZ/TEZ/textbook_bert_embeddings_Trust.npy'

textbook_embeddings = np.load(textbook_embeddings_path)
required_shape = 165

n_samples_textbook = textbook_embeddings.shape[0]
if textbook_embeddings.shape[1] < required_shape:
    pad_width = required_shape - textbook_embeddings.shape[1]
    textbook_embeddings_padded = np.pad(textbook_embeddings, ((0, 0), (0, pad_width), (0, 0)), mode='constant')
else:
    textbook_embeddings_padded = textbook_embeddings[:, :required_shape, :]

textbook_embeddings_reshaped = textbook_embeddings_padded.reshape((n_samples_textbook, required_shape, 768))

# Predict using the model
y_pred_textbook_scores = model.predict(textbook_embeddings_reshaped).flatten()



# Load sentences for reference
#textbook_path = '/Users/eceserin/Desktop/BAOR/TEZ/TEZ/processed_sentences_His Family.txt'
textbook_path = '/Users/eceserin/Desktop/BAOR/TEZ/TEZ/processed_sentences_Trust.txt'
with open(textbook_path, 'r', encoding='utf-8') as file:
    textbook_sentences = [sentence.strip() for sentence in file if sentence.strip()]

# Debugging check for sentence and score alignment
print(f"Number of sentences in textbook: {len(textbook_sentences)}")
print(f"Number of prediction scores: {len(y_pred_textbook_scores_scaled)}")

# Proceed if the lengths match
if len(textbook_sentences) == len(y_pred_textbook_scores_scaled):
    # Calculate inclusivity score
    non_sexist_count = np.sum(y_pred_textbook_scores_scaled < 0.5)
    total_sentences = len(y_pred_textbook_scores_scaled)
    non_sexist_percentage = (non_sexist_count / total_sentences) * 100
    print(f"Percentage of Non-Sexist Sentences in the textbook: {non_sexist_percentage:.2f}%")

    # Print the top 5 sentences with highest and lowest scaled scores
    top_5_highest = np.argsort(y_pred_textbook_scores_scaled)[-15:][::-1]
    print("\nTop 5 Most Sexist Predictions (Highest Scores):")
    for idx in top_5_highest:
        print(f"Sentence: {textbook_sentences[idx]}\nScore: {y_pred_textbook_scores_scaled[idx]}\n")

    top_5_lowest = np.argsort(y_pred_textbook_scores_scaled)[:15]
    print("\nTop 5 Least Sexist Predictions (Lowest Scores):")
    for idx in top_5_lowest:
        print(f"Sentence: {textbook_sentences[idx]}\nScore: {y_pred_textbook_scores_scaled[idx]}\n")

    # Filter and print top 5 least sexist sentences with target words
    target_words = ["man", "men", "woman", "women", "he", "she", "girl", "boy", "girls", "boys"]
    filtered_least_sexist = pd.DataFrame({
        'Sentence': textbook_sentences,
        'Score': y_pred_textbook_scores_scaled
    }).sort_values(by='Score').reset_index(drop=True)

    filtered_least_sexist = filtered_least_sexist[filtered_least_sexist['Score'] <= 0.5]
    filtered_least_sexist['Contains Target Word'] = filtered_least_sexist['Sentence'].apply(
        lambda sentence: any(word in sentence.lower().split() for word in target_words)
    )
    filtered_least_sexist_with_target = filtered_least_sexist[filtered_least_sexist['Contains Target Word']]

    top_5_least_sexist_with_target = filtered_least_sexist_with_target.nsmallest(15, 'Score')
    print("\nTop 5 Least Sexist Sentences Containing Target Words:")
    for idx, row in top_5_least_sexist_with_target.iterrows():
        print(f"Sentence: {row['Sentence']}\nScore: {row['Score']}\n")



    # Plot the distribution of scaled prediction scores
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(y_pred_textbook_scores_scaled, bins=20, color='gray', edgecolor='black')

    for i, patch in enumerate(patches):
        # Check if the center of the bin is greater than or equal to 0.5 for red color
        bin_center = (bins[i] + bins[i + 1]) / 2
        if bin_center >= 0.5:
            patch.set_facecolor('red')
        else:
            patch.set_facecolor('green')

    plt.xlabel('Prediction Score (Weight)')
    plt.ylabel('Count')
    plt.title('Distribution of Prediction Scores in Trust')
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.show()

else:
    print("Mismatch in the number of sentences and prediction scores. Please check the data alignment.")'''
    
    
    
    
    
#TEST FOR OTHER SENTENCES
from sklearn.preprocessing import MinMaxScaler
import numpy as np


textbook_embeddings_path = '/Users/eceserin/Desktop/BAOR/TEZ/TEZ/where does agent 3 fail.npy'
textbook_embeddings = np.load(textbook_embeddings_path)
required_shape = 165


n_samples_textbook = textbook_embeddings.shape[0]
if textbook_embeddings.shape[1] < required_shape:
    pad_width = required_shape - textbook_embeddings.shape[1]
    textbook_embeddings_padded = np.pad(textbook_embeddings, ((0, 0), (0, pad_width), (0, 0)), mode='constant')
else:
    textbook_embeddings_padded = textbook_embeddings[:, :required_shape, :]

textbook_embeddings_reshaped = textbook_embeddings_padded.reshape((n_samples_textbook, required_shape, 768))

# Predict using the model
y_pred_textbook_scores = model.predict(textbook_embeddings_reshaped).flatten()


# Load sentences for reference
textbook_path = '/Users/eceserin/Desktop/BAOR/TEZ/TEZ/Where does agent 1 break.txt'
with open(textbook_path, 'r', encoding='utf-8') as file:
    textbook_sentences = [sentence.strip() for sentence in file if sentence.strip()]

# Debugging check for sentence and score alignment
print(f"Number of sentences in textbook: {len(textbook_sentences)}")
print(f"Number of prediction scores: {len(y_pred_textbook_scores)}") 

# Ensure the lengths match before proceeding
if len(textbook_sentences) == len(y_pred_textbook_scores):
    # Calculate inclusivity score
    non_sexist_count = np.sum(y_pred_textbook_scores < 0.5) 
    total_sentences = len(y_pred_textbook_scores)
    non_sexist_percentage = (non_sexist_count / total_sentences) * 100
    print(f"Percentage of Non-Sexist Sentences: {non_sexist_percentage:.2f}%\n")

    # Print all sentences with their respective scores
    print("All Sentences with their Respective Scores:\n")
    for sentence, score in zip(textbook_sentences, y_pred_textbook_scores):
        print(f"Sentence: {sentence}\nScore: {score:.3f}\n")
else:
    print("Mismatch in the number of sentences and prediction scores. Please check the data alignment.")
