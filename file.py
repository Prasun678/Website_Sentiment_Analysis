#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install requests beautifulsoup4 nltk


# In[2]:


import nltk
nltk.download('vader_lexicon')


# In[13]:


import pandas as pd
import os
import re
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

# Download NLTK resources
nltk.download('punkt')

# Function to extract text from URL
def extract_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    # Extract text from <p> tags or any other suitable tag based on the webpage structure
    paragraphs = soup.find_all('p')
    text = ' '.join([p.get_text() for p in paragraphs])
    return text

# Function to clean text using stop words
def clean_text(text, stop_words):
    cleaned_words = [word.lower() for word in word_tokenize(text) if word.isalnum() and word.lower() not in stop_words]
    return ' '.join(cleaned_words)

# Function to calculate sentimental analysis metrics
def sentimental_analysis(text, positive_words, negative_words):
    tokens = word_tokenize(text.lower())
    positive_score = sum(1 for word in tokens if word in positive_words)
    negative_score = sum(1 for word in tokens if word in negative_words)
    polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)
    subjectivity_score = (positive_score + negative_score) / (len(tokens) + 0.000001)
    return positive_score, negative_score, polarity_score, subjectivity_score

# Function to count syllables in a word
def syllable_count(word):
    word = word.lower()
    count = 0
    vowels = 'aeiouy'
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith('e'):
        count -= 1
    if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
        count += 1
    if count == 0:
        count += 1
    return count

# Function to calculate readability metrics
def readability_analysis(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    total_words = len(words)
    total_sentences = len(sentences)
    average_words_per_sentence = total_words / total_sentences
    complex_words = [word for word in words if syllable_count(word) > 2]
    complex_word_count = len(complex_words)
    percentage_complex_words = (complex_word_count / total_words) * 100
    fog_index = 0.4 * (average_words_per_sentence + percentage_complex_words)
    return average_words_per_sentence, percentage_complex_words, fog_index

# Function to count personal pronouns
def personal_pronouns_count(text):
    pronouns = re.findall(r'\b(?:I|we|my|ours|us)\b', text)
    return len(pronouns)

# Function to calculate average word length
def average_word_length(text):
    words = word_tokenize(text.lower())
    total_characters = sum(len(word) for word in words)
    return total_characters / len(words)

# Function to count syllables per word
def syllables_per_word(text):
    words = word_tokenize(text.lower())
    syllable_count_per_word = [syllable_count(word) for word in words]
    return sum(syllable_count_per_word) / len(words)

# Read stop words from various files
def read_stopwords(folder_path):
    stopwords_combined = set()
    for file_name in os.listdir(folder_path):
        with open(os.path.join(folder_path, file_name), 'r') as file:
            stopwords_combined.update(file.read().splitlines())
    return stopwords_combined

# Load positive and negative words
positive_words = set(open("positive-words.txt").read().splitlines())
negative_words = set(open('negative-words.txt').read().splitlines()) 

# Read stopwords from folder
stopwords_folder = 'Stop_word'
stop_words = read_stopwords(stopwords_folder)

# Read URLs from Excel file
df = pd.read_excel(r"E:\New folder\Input.xlsx")

# Initialize lists to store results
sentimental_results = []
readability_results = []
other_metrics = []

# Process each URL
for url in df['URL']:
    text = extract_text_from_url(url)
    cleaned_text = clean_text(text, stop_words)
    
    # Sentimental analysis
    positive_score, negative_score, polarity_score, subjectivity_score = sentimental_analysis(cleaned_text, positive_words, negative_words)
    
    # Readability analysis
    average_words_per_sentence, percentage_complex_words, fog_index = readability_analysis(cleaned_text)
    
    # Other metrics
    word_count = len(word_tokenize(cleaned_text))
    personal_pronouns = personal_pronouns_count(text)
    average_length = average_word_length(cleaned_text)
    syllables_per_word_value = syllables_per_word(cleaned_text)
    avg_number_of_words_per_sentence = len(word_tokenize(text)) / len(sent_tokenize(text))
    complex_word_count = len([word for word in word_tokenize(text.lower()) if syllable_count(word) > 2])
    
    sentimental_results.append([positive_score, negative_score, polarity_score, subjectivity_score])
    readability_results.append([average_words_per_sentence, percentage_complex_words, fog_index])
    other_metrics.append([word_count, personal_pronouns, average_length, avg_number_of_words_per_sentence,
                          complex_word_count, syllables_per_word_value])

# Create DataFrames for each result
sentimental_df = pd.DataFrame(sentimental_results, columns=['POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE', 'SUBJECTIVITY SCORE'])
readability_df = pd.DataFrame(readability_results, columns=['AVG SENTENCE LENGTH', 'PERCENTAGE OF COMPLEX WORDS', 'FOG INDEX'])
other_metrics_df = pd.DataFrame(other_metrics, columns=['WORD COUNT', 'PERSONAL PRONOUNS', 'AVG WORD LENGTH',
                                                        'AVG NUMBER OF WORDS PER SENTENCE', 'COMPLEX WORD COUNT', 'SYLLABLES PER WORD'])

# Concatenate all DataFrames
results_df = pd.concat([sentimental_df, readability_df, other_metrics_df], axis=1)

# Write results to Excel file
with pd.ExcelWriter('output.xlsx', engine='openpyxl') as writer:
    results_df.to_excel(writer, index=False)


# In[ ]:




