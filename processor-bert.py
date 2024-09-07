import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from bertopic import BERTopic
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download necessary NLTK resources
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()


# Function to clean text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    return text


# Tokenize text
def tokenize_text(text):
    return word_tokenize(text)


# Remove stopwords
stop_words = set(stopwords.words('english'))


def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]


# Lemmatize tokens
def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]


# Remove non-alphabetic characters
def remove_non_alpha(tokens):
    return [word for word in tokens if word.isalpha()]


# Full preprocessing function
def preprocess_text(text):
    text = clean_text(text)
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)
    tokens = remove_non_alpha(tokens)
    return ' '.join(tokens)


# Load your data
csv_file_path = os.path.join('results', '8_final_llm_filtered_results.csv')
output_directory = 'results'

df = pd.read_csv(csv_file_path)
logging.info("CSV file loaded successfully.")

# Filter rows with non-string abstracts and titles
df = df[df['Abstract'].apply(lambda x: isinstance(x, str))]
df = df[df['Title'].apply(lambda x: isinstance(x, str))]
logging.info("Filtered non-string abstracts and titles.")

# Preprocess 'Abstract' and 'Title' columns
df['Processed_Abstract'] = df['Abstract'].apply(preprocess_text)
df['Processed_Title'] = df['Title'].apply(preprocess_text)
logging.info("Preprocessing completed for abstracts and titles.")

# Combine preprocessed 'Title' and 'Abstract' for topic modeling
df['Text'] = df['Processed_Title'] + " " + df['Processed_Abstract']

# Initialize BERTopic model
topic_model = BERTopic()

# Fit BERTopic model on the preprocessed text data
topics, probabilities = topic_model.fit_transform(df['Text'].tolist())

# Add the discovered topics to your DataFrame
df['Topic'] = topics

# Log all topics found by BERTopic
topic_info = topic_model.get_topic_info()
logging.info("Topics discovered by BERTopic:")
for i, row in topic_info.iterrows():
    logging.info(f"Topic {row['Topic']}: {row['Name']} (Size: {row['Count']})")

# Initialize a dictionary to log relevant topics for each journal
relevant_topics_per_journal = {index: [] for index in df.index}

# Map topics to each journal and log them
for index, row in df.iterrows():
    journal_topics = topics[index]
    topic_words = topic_model.get_topic(journal_topics)
    relevant_topics_per_journal[index] = topic_words

    logging.info(f"Journal {index + 1} ({df.loc[index, 'Title']}): Relevant topics: {topic_words}")

# Save the final DataFrame with topic indicators to a file
output_file_path_df = os.path.join(output_directory, 'final_dataframe_with_topics_bertopic.csv')
df.to_csv(output_file_path_df, index=False)
logging.info(f"DataFrame with topic indicators saved to {output_file_path_df}")

# Save the final list of unique topics and their counts to a file
output_file_path_counts = os.path.join(output_directory, 'final_topics_list_with_counts.txt')
with open(output_file_path_counts, 'w') as f:
    for i, row in topic_info.iterrows():
        f.write(f"Topic {row['Topic']}: {row['Name']} (Size: {row['Count']})\n")
logging.info(f"Topic counts saved to {output_file_path_counts}")
