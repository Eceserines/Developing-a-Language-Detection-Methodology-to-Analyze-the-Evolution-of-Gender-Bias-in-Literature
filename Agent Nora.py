#USE ENV_CNN_CLEAN

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt

# Load the dataset
dataset_path = '/Users/eceserin/Desktop/BAOR/TEZ/TEZ/sexism_data.csv'
sexism_data = pd.read_csv(dataset_path)
y = sexism_data['sexist'].apply(lambda x: 1 if x == True else 0).values  
sentences = sexism_data['text'].values  # Store the sentences

# Vectorize the sentences using a Bag-of-Words (binary) approach
vectorizer = CountVectorizer(binary=True) 
X = vectorizer.fit_transform(sentences).toarray()  
print(f"Bag-of-Words matrix shape: {X.shape}")

# Split into train and test sets
X_train, X_test, y_train, y_test, sentences_train, sentences_test = train_test_split(X, y, sentences, test_size=0.2, random_state=42)



# Build a simple Dense Neural Network model 
def build_dense_nn_model(learning_rate):
    model = Sequential()

    # First fully connected layer with Batch Normalization
    model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(BatchNormalization()) 
    model.add(Dropout(0.3))
    
    # Second fully connected layer
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())  
    model.add(Dropout(0.3))
    
    # Final output layer using sigmoid for binary classification
    model.add(Dense(1, activation='sigmoid'))  # Sigmoid activation for binary classification
    
    # Compile the model with Adam optimizer and binary_crossentropy loss
    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Train the Dense Neural Network model for binary classification
learning_rate = 0.00005
model = build_dense_nn_model(learning_rate)
history = model.fit(X_train, y_train, epochs=9, batch_size=32, validation_data=(X_test, y_test), verbose=1)





# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy}')

# Predict on the test set
y_pred_test = (model.predict(X_test).flatten() > 0.5).astype(int)  

# Save the predicted scores and sentences to an Excel file
data_all = {'Tested Sentence': sentences_test, 'True Label': y_test, 'Predicted Label': y_pred_test}
df_all = pd.DataFrame(data_all)
output_path_all = '/Users/eceserin/Desktop/BAOR/TEZ/TEZ/cnn_test_results_agent_1.xlsx'
df_all.to_excel(output_path_all, index=False)
print(f"Test results (sentences, true labels, and predicted labels) saved to {output_path_all}")



'''# Calculate and print the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_test)
print("Confusion Matrix:")
print(conf_matrix)

# Plot the confusion matrix using matplotlib
plt.figure(figsize=(6, 4))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Dense NN with Bag-of-Words')
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
Precision-Recall curve
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



#Plot training and validation accuracy over epochs
plt.figure(figsize=(8, 6))
plt.plot(np.arange(1, len(history.history['acc'])+1), history.history['acc'], label='Train Accuracy')  # Adjust x-axis
plt.plot(np.arange(1, len(history.history['val_acc'])+1), history.history['val_acc'], label='Validation Accuracy')  # Adjust x-axis
plt.title('Model Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
'''



#TESTING

# Load the textbook data
#textbook_path = '/Users/eceserin/Desktop/BAOR/TEZ/TEZ/processed_sentences_His Family.txt'
#textbook_path = '/Users/eceserin/Desktop/BAOR/TEZ/TEZ/processed_sentences_Trust.txt'
#textbook_path = '/Users/eceserin/Desktop/BAOR/TEZ/TEZ/Sentences from Agent 123.txt'
textbook_path = '/Users/eceserin/Desktop/BAOR/TEZ/TEZ/Where does agent 1 break.txt'




with open(textbook_path, 'r', encoding='utf-8') as file:
    textbook_data = file.readlines()

# Clean up and prepare the textbook data
textbook_sentences = [sentence.strip() for sentence in textbook_data if sentence.strip()]
print(f"Number of textbook sentences: {len(textbook_sentences)}")

# Vectorize the textbook sentences using the vocabulary from the sexism_data
def vectorize_new_data(sentences, vectorizer):
    vectorized = np.zeros((len(sentences), len(vectorizer.vocabulary_)))  
    for i, sentence in enumerate(sentences):
        tokens = sentence.split()
        for token in tokens:
            if token in vectorizer.vocabulary_:  # Only use tokens that exist in the original training data
                vectorized[i, vectorizer.vocabulary_[token]] = 1
    return vectorized

X_textbook = vectorize_new_data(textbook_sentences, vectorizer)
print(f"Textbook data matrix shape: {X_textbook.shape}")


X_textbook_scaled = X_textbook

# Predict on the textbook data using the trained model
y_pred_weights = model.predict(X_textbook_scaled).flatten()

# Binarize the predictions 
y_pred_textbook = (y_pred_weights > 0.5).astype(int)

# Save the predictions along with the textbook sentences and weights to an Excel file
data_all_textbook = {
    'Tested Sentence': textbook_sentences, 
    'Predicted Label': y_pred_textbook,
    'Prediction Score (Weight)': y_pred_weights
}
df_all_textbook = pd.DataFrame(data_all_textbook)
output_path_textbook = '/Users/eceserin/Desktop/BAOR/TEZ/TEZ/where does agent 1 break.xlsx'
df_all_textbook.to_excel(output_path_textbook, index=False)
print(f"Test results (textbook sentences, predicted labels, and weights) saved to {output_path_textbook}")

# Calculate the percentage of non-sexist sentences
non_sexist_count = np.sum(y_pred_textbook == 0)
total_sentences = len(y_pred_textbook)
non_sexist_percentage = (non_sexist_count / total_sentences) * 100
print(f"Percentage of Non-Sexist Sentences: {non_sexist_percentage:.2f}%")




# Find and print the top 5 sentences with the highest weights (most sexist predictions)
top_5_highest = df_all_textbook.nlargest(10, 'Prediction Score (Weight)')
print("\nTop 5 Sentences with the Highest Weights (Most Sexist Predictions):")
for idx, row in top_5_highest.iterrows():
    print(f"{idx+1}. Sentence: {row['Tested Sentence']}\n   Weight: {row['Prediction Score (Weight)']}\n")
    
    
'''
# Find and print the top 5 sentences with the lowest weights (most non-sexist predictions)
top_5_lowest = df_all_textbook.nsmallest(5, 'Prediction Score (Weight)')
print("\nTop 5 Sentences with the Lowest Weights (Most Non-Sexist Predictions):")
for idx, row in top_5_lowest.iterrows():
    print(f"{idx+1}. Sentence: {row['Tested Sentence']}\n   Weight: {row['Prediction Score (Weight)']}\n")
    

# Define the list of target words
target_words = ["man", "men", "woman", "women", "he", "she", "girl", "boy", "girls", "boys"]

# Filter and print the top 5 least sexist sentences that contain any target word
filtered_least_sexist = df_all_textbook[df_all_textbook['Prediction Score (Weight)'] <= 0.5]  # Only non-sexist predictions
filtered_least_sexist['Contains Target Word'] = filtered_least_sexist['Tested Sentence'].apply(
    lambda sentence: any(word in sentence.lower().split() for word in target_words)
)

# Select only rows where at least one target word is present
filtered_least_sexist_with_target = filtered_least_sexist[filtered_least_sexist['Contains Target Word']]

# Get the top 5 least sexist sentences with target words
top_5_least_sexist_with_target = filtered_least_sexist_with_target.nsmallest(5, 'Prediction Score (Weight)')

# Print the results
print("\nTop 5 Least Sexist Sentences Containing Target Words:")
for idx, row in top_5_least_sexist_with_target.iterrows():
    print(f"{idx+1}. Sentence: {row['Tested Sentence']}\n   Weight: {row['Prediction Score (Weight)']}\n")
    



# Plot the distribution of prediction scores (weights) as a histogram
plt.figure(figsize=(10, 6))
n, bins, patches = plt.hist(y_pred_weights, bins=20, color='gray', edgecolor='black')

# Set color based on the bin center
for i, patch in enumerate(patches):
    if bins[i] >= 0.5:  # Set red color for bins with center >= 0.5
        patch.set_facecolor('red')
    else:  # Set green color for bins with center < 0.5
        patch.set_facecolor('green')

# Add labels and title
plt.xlabel('Prediction Score (Weight)')
plt.ylabel('Count')  # Changed from 'Density' to 'Count'
plt.title('Distribution of Prediction Scores in His Family')
plt.xticks(np.arange(0, 1.1, 0.1))  # Set x-axis ticks at intervals of 0.1

# Show the plot
plt.show()
'''