#USE ENV_KOD

import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
import ssl
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re


nltk.data.path.clear()
nltk.data.path.append('/Users/eceserin/nltk_data')

# Workaround for SSL certificate verification
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Define the path to the text file
#file_path = '/Users/eceserin/Desktop/BAOR/TEZ/TEZ/Hernan Diaz_Trust.txt'
file_path = '/Users/eceserin/Desktop/BAOR/TEZ/TEZ/Ernest Poole_His Family.txt'


# Read the content of the file
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

# Convert the text to lowercase
text = text.lower()

# List of common abbreviations with period
abbreviations = [
    "mr.", "mrs.", "ms.", "dr.", "jr.", "sr.", "prof.", "st.", "i.e.", "e.g.", "vs.", "inc.", "ltd.", 
    "co.", "corp.", "ave.", "blvd.", "rd.", "ln.", "mt.", "ft.", "capt.", "sgt.", "lt.", "col.", "maj.",
    "gen.", "rep.", "sen.", "gov.", "pres.", "atty.", "supt.", "det.", "rev.", "hon.", "cmdr.", 
    "treas.", "sec.", "amb.", "ph.d.", "m.d.", "b.a.", "m.a.", "d.d.s.", "r.n.", "esq.", "univ."
]

# Split the text into sentences based on punctuation
temp_sentences = re.split(r'(?<=[.!?])\s+', text)

# Rejoin sentences split by abbreviations
sentences = []
for i in range(len(temp_sentences)):
    if i > 0 and any(temp_sentences[i-1].strip().endswith(abbrev) for abbrev in abbreviations):
        sentences[-1] += " " + temp_sentences[i]
    else:
        sentences.append(temp_sentences[i])

# Display sample sentences to check
print(f"Sample sentences: {sentences[:5]}")

# Calculate sentence statistics
total_sentences = len(sentences)
total_words = sum(len(word_tokenize(sentence)) for sentence in sentences)
average_words_per_sentence = total_words / total_sentences if total_sentences > 0 else 0

print(f"Total number of sentences: {total_sentences}")
print(f"Total number of words: {total_words}")
print(f"Average number of words per sentence: {average_words_per_sentence:.2f}")

# Tokenize the text at the word level
tokens = word_tokenize(text)

# Define punctuation to remove, excluding commas and periods
punctuation = list(string.punctuation)
punctuation.remove(',')
punctuation.remove('.')
extra_punctuation = ['``', "''", '“', '”', '–','* * * * *', '—', '‘', '’', '--']
punctuation.extend(extra_punctuation)

# Remove all unwanted punctuation from the tokens
tokens = [token for token in tokens if token not in punctuation]

# Get the list of stopwords, excluding "no", "not", "she", and "he"
stop_words = set(stopwords.words('english')) - {'no', 'not', 'she', 'he'}
stop_words.update(["'s", "'d", "'ll", "'ve", "'re", "n't", "'m", "would", "could", "* * * * *", "chapter"])

# POS tagging for extracting relevant parts of speech
tagged_tokens = nltk.pos_tag(tokens)
adj_noun_tokens = [(word, tag) for word, tag in tagged_tokens if tag in ('JJ', 'NN', 'NNS')]

# Remove stopwords from the tokens
tokens = [token for token in tokens if token.lower() not in stop_words]

# Stemming
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(token) for token in tokens]

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

# Display the first 50 tokens after lemmatization
print("First 50 tokens after lemmatization:", lemmatized_tokens[:50])

# Save the preprocessed text (lemmatized tokens) as a new text file
preprocessed_text_path = '/Users/eceserin/Desktop/BAOR/TEZ/TEZ/preprocessed_text.txt'
with open(preprocessed_text_path, 'w', encoding='utf-8') as file:
    file.write(' '.join(lemmatized_tokens))

# Save the processed full sentences for BERT processing
sentences_path = '/Users/eceserin/Desktop/BAOR/TEZ/TEZ/processed_sentences_His Family.txt'
with open(sentences_path, 'w', encoding='utf-8') as file:
    file.write('\n'.join(sentences))

print(f"Sentences saved to {sentences_path}")

#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################

'''
# Now we focus on trigram sentence segmentation as mentioned in the methodology

# Function to generate n-grams
def generate_ngrams(tokens, n):
    n_grams = list(ngrams(tokens, n))
    return [' '.join(grams) for grams in n_grams]

# Generate bigrams and trigrams from the tokenized sentences
bigrams = generate_ngrams(lemmatized_tokens, 2)
trigrams = generate_ngrams(lemmatized_tokens, 3)

# Generate trigrams from the tokenized sentences (new addition)
trigram_list = generate_ngrams(lemmatized_tokens, 3)

# Display the first 20 trigrams
print("\nFirst 20 trigrams:")
print(trigram_list[:20])

# Save the trigrams (keeping your original file-saving logic)
trigrams_path = '/Users/eceserin/Desktop/BAOR/TEZ/TEZ/trigrams_Trust.txt'
with open(trigrams_path, 'w', encoding='utf-8') as file:
    for trigram in trigrams:
        file.write(trigram + '\n')

# Now, we also update the longest sentence finder to use punctuation correctly
def find_longest_sentence(sentences):
    longest_sentence = ''
    max_word_count = 0
    
    for sentence in sentences:
        word_count = len(word_tokenize(sentence))
        if word_count > max_word_count:
            max_word_count = word_count
            longest_sentence = sentence
            
    return longest_sentence, max_word_count

# Find the longest sentence based on word count
longest_sentence, word_count = find_longest_sentence(sentences)

print(f"The longest sentence is:\n{longest_sentence}")
print(f"\nNumber of words in the longest sentence: {word_count}")





'''

'''
#FROM HERE ON, I START THE WORD EMBEDDINGS
from gensim.models import Word2Vec


# Modify tokens to mark gendered context explicitly
# Find gender-specific tokens
he_context = [index for index, word in enumerate(lemmatized_tokens) if word == 'he']
she_context = [index for index, word in enumerate(lemmatized_tokens) if word == 'she']

# Identify nearby words (within a window size of 5) around gendered tokens
def context_window(tokens, index, window=5):
    start = max(index - window, 0)
    end = min(index + window + 1, len(tokens))
    return tokens[start:end]

# Add proximity markers for gendered words
for index in he_context:
    context_words = context_window(lemmatized_tokens, index)
    lemmatized_tokens.extend([f"He_{word}" for word in context_words])

for index in she_context:
    context_words = context_window(lemmatized_tokens, index)
    lemmatized_tokens.extend([f"She_{word}" for word in context_words])

# Function to categorize nouns 
def categorize_noun(word, pos_tag):
    if pos_tag == 'NNP':  # Proper Noun
        return 'P' if word.endswith('s') else 'S'
    elif pos_tag == 'NN':  # Common Noun
        return 'C' if word.endswith('s') else 'S'
    return None

# Function to categorize adjectives
def categorize_adjective(word, pos_tag):
    if pos_tag == 'JJ':  # Qualitative Adjective
        return 'Q'
    elif pos_tag == 'JJS':  # Superlative Adjective
        return 'S'
    elif pos_tag == 'JJR':  # Comparative Adjective
        return 'C'
    return None

# Tagging nouns and adjectives in the token list
noun_tags = [(word, categorize_noun(word, tag)) for word, tag in tagged_tokens if tag.startswith('NN')]
adj_tags = [(word, categorize_adjective(word, tag)) for word, tag in tagged_tokens if tag.startswith('JJ')]

# Combine into a single vector representation
word_embeddings = []

for word, tag in noun_tags + adj_tags:
    word_embeddings.append((word, tag))  # Store the word and its tag

# Train Word2Vec on modified lemmatized tokens with gender context
word2vec_model = Word2Vec(sentences=[lemmatized_tokens], vector_size=100, window=5, min_count=1, workers=4)

# Save the updated model
word2vec_model_path = '/Users/eceserin/Desktop/BAOR/TEZ/TEZ/word2vec_model_gender_context.model'
word2vec_model.save(word2vec_model_path)


'''

'''
    
#BASIC ANALYSIS STARTS HERE#
   
from collections import Counter

# Count the number of different words in the text
unique_words = set(lemmatized_tokens)
num_unique_words = len(unique_words)
print(f"Number of different words in the text: {num_unique_words}")

# Find the most frequently repeated words
word_freq = Counter(lemmatized_tokens)
most_common_words = word_freq.most_common(10) 
print("\nMost frequently repeated words:")
for word, freq in most_common_words:
    print(f"{word}: {freq}")

# Find the most frequently repeated bi-grams
bigram_freq = Counter(bigrams)
most_common_bigrams = bigram_freq.most_common(20)  
print("\nMost frequently repeated bi-grams:")
for bigram, freq in most_common_bigrams:
    print(f"{bigram}: {freq}")

# Find the most frequently repeated tri-grams
trigram_freq = Counter(trigrams)
most_common_trigrams = trigram_freq.most_common(10) 
print("\nMost frequently repeated tri-grams:")
for trigram, freq in most_common_trigrams:
    print(f"{trigram}: {freq}")


#BEFORE AND AFTER S/HE

from collections import Counter

# Function to find the most common words before and after a specific word
def find_common_words(tokens, target_word):
    before_words = []
    after_words = []

    for i, word in enumerate(tokens):
        if word == target_word:
            if i > 0:
                before_words.append(tokens[i-1])
            if i < len(tokens) - 1:
                after_words.append(tokens[i+1])

    most_common_before = Counter(before_words).most_common(3)
    most_common_after = Counter(after_words).most_common(3)

    return most_common_before, most_common_after

# Find the most common words before and after "he" and "she"
he_before, he_after = find_common_words(lemmatized_tokens, 'he')
she_before, she_after = find_common_words(lemmatized_tokens, 'she')

print(f"Most common words before 'he': {he_before}")
print(f"Most common words after 'he': {he_after}")
print(f"Most common words before 'she': {she_before}")
print(f"Most common words after 'she': {she_after}")






'''

###WORD CLOUD CREATING AND THE CONTINUATION OF THE BASIC ANALYSIS
from nltk import pos_tag
from collections import Counter
import csv

# Function to find adjectives associated with gender-related words within a window
def find_adj_near_gendered_terms(tokens, gender_terms, window_size=5):
    adjectives = []
    for i, word in enumerate(tokens):
        if word in gender_terms:
            # Define the window around the gendered term
            start = max(i - window_size, 0)
            end = min(i + window_size + 1, len(tokens))
            
            # Check words within the window
            for j in range(start, end):
                if j != i:  # Exclude the gendered term itself
                    pos_tag_word = pos_tag([tokens[j]])[0][1]
                    if pos_tag_word == 'JJ':  # Adjective POS tag
                        adjectives.append(tokens[j])
    return adjectives

# List of gender-related terms
gender_terms = ['he', 'she', 'him', 'her', 'man', 'woman', 'men', 'women', 'boy', 'boys', 'girl', 'girls']

# Find adjectives associated with gender terms
adj_near_gendered_terms = find_adj_near_gendered_terms(lemmatized_tokens, gender_terms)

# Count the frequency of each adjective
adj_freq = Counter(adj_near_gendered_terms)

# Separate frequencies by gender for output to CSV
he_she_adj_freq = {word: freq for word, freq in adj_freq.items() if word in ['he', 'him', 'man', 'men', 'boy', 'boys']}
she_adj_freq = {word: freq for word, freq in adj_freq.items() if word in ['she', 'her', 'woman', 'women', 'girl', 'girls']}

# Save adjectives and their frequencies to CSV files
with open('/Users/eceserin/Desktop/BAOR/TEZ/TEZ/he_adj_near_gendered_terms_wordcloud.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    for word, freq in he_she_adj_freq.items():
        writer.writerow([freq, word])

with open('/Users/eceserin/Desktop/BAOR/TEZ/TEZ/she_adj_near_gendered_terms_wordcloud.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    for word, freq in she_adj_freq.items():
        writer.writerow([freq, word])


'''
#verbs
import csv
from collections import Counter
import nltk


nltk.download('averaged_perceptron_tagger')

def find_verbs_for_pronoun(tokens, pronoun):
    verbs = []
    for i, word in enumerate(tokens):
        if word == pronoun and i < len(tokens) - 1:
            pos_tag = nltk.pos_tag([tokens[i+1]])[0][1]  # Get the POS tag of the word after the pronoun
            if pos_tag.startswith('VB'):  # Verb POS tags start with 'VB'
                verbs.append(tokens[i+1])
    return verbs

# Find verbs for "he" and "she"
he_verbs = find_verbs_for_pronoun(lemmatized_tokens, 'he')
she_verbs = find_verbs_for_pronoun(lemmatized_tokens, 'she')

# Count the frequency of each verb
he_verbs_freq = Counter(he_verbs)
she_verbs_freq = Counter(she_verbs)

# Save the verbs and their frequencies to CSV files in the correct format for word clouds
with open('/Users/eceserin/Desktop/BAOR/TEZ/TEZ/he_verbs_wordcloud.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    for word, freq in he_verbs_freq.items():
        writer.writerow([freq, word])

with open('/Users/eceserin/Desktop/BAOR/TEZ/TEZ/she_verbs_wordcloud.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    for word, freq in she_verbs_freq.items():
        writer.writerow([freq, word])
        
'''