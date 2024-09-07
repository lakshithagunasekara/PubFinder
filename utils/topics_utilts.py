import json
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import logging
from openai import OpenAI
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
import json
import os

import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from openai import OpenAI
import logging
from pydantic import BaseModel, Field
from typing import List
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE

import config

client = OpenAI(api_key=config.openai_key)

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()


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


def preprocess_topic(topic):
    topic = topic.lower()  # Convert to lowercase
    topic = re.sub(r'[^\w\s]', '', topic)  # Remove special characters
    topic = ' '.join([lemmatizer.lemmatize(word) for word in topic.split()])  # Lemmatization
    return topic.strip()


class JournalResponse(BaseModel):
    journal_id: int
    topics: List[str]


class GPTResponse(BaseModel):
    journals: List[JournalResponse]


def get_relevant_topics_gpt_batch(journals_batch):
    prompt = (
        f"Please read the following title and abstract pairs, and return 3 topics each for a pair that are relevant, "
        f"meaningful and provides a good understanding of the journal content. "
        f"As these topics are for a systematic review of Responsible Generative AI, make sure the topics are well computed, "
        f"Return the result as a structured JSON format with 'journal_id', 'title', 'abstract', and 'topics'. "
        "Each journal should have 1 to 3 relevant topics. Return the topics in a list of strings."
        "Do not use generic topics such as responsible AI, Generative AI, computer science. "
        "Always try to use meaningful topics which can explain the journal\n\n"
    )

    for i, journal in enumerate(journals_batch, start=1):
        title = journal['Title']
        abstract = journal['Abstract']
        prompt += f"Journal {i}:\nTitle: {title}\nAbstract: {abstract}\n\n"

    response = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "You are an AI that helps identify key topics from research papers."},
            {"role": "user", "content": prompt}
        ],
        response_format=GPTResponse
    )

    return response.choices[0].message.parsed


def extract_topics(response_text):
    new_topics = [topic.strip() for topic in response_text.split(",")]
    if new_topics:
        return new_topics
    else:
        logging.warning("No topics found in the GPT response.")
        return []


# Function to update the master topic list with preprocessing and count occurrences
def update_topic_list(new_topics, topic_counts):
    for topic in new_topics:
        if topic not in topic_counts:
            topic_counts[topic] = 1  # Initialize count
            logging.info(f"New topic added: {topic}")
        else:
            topic_counts[topic] += 1  # Increment count
    return topic_counts


def get_topics_for_journals_df(df, output_directory):
    batch_size = 5
    topic_counts = {}
    df_file_path = os.path.join(output_directory, '11_journal_data_with_topics.csv')
    topic_counts_file = os.path.join('results', '12_topics_list_with_counts.txt')
    if config.reuse["topics"]:
        df = pd.read_csv(df_file_path)
        with open(topic_counts_file, 'r') as f:
            for line in f:
                topic, count = line.strip().split(': ')
                topic_counts[topic] = int(count)

    else:
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i + batch_size]

            # Prepare batch data for GPT
            journals_batch = [
                {"Title": row['Processed_Title'], "Abstract": row['Processed_Abstract']}
                for _, row in batch_df.iterrows()
            ]

            # Get topics for the batch
            gpt_response = get_relevant_topics_gpt_batch(journals_batch)

            # Process and store topics for each journal
            for journal_response in gpt_response.journals:
                journal_id = i + journal_response.journal_id - 1
                processed_topics = [preprocess_topic(topic) for topic in journal_response.topics]
                df.at[journal_id, 'Topics'] = ', '.join(processed_topics)
                topic_counts = update_topic_list(processed_topics, topic_counts)

        df.to_csv(df_file_path, index=False)
        logging.info(f"Journals with Topics saved to {output_directory}/11_journal_data_with_topics.csv")
        with open(topic_counts_file, 'w') as f:
            for topic, count in topic_counts.items():
                f.write(f"{topic}: {count}\n")
        logging.info(f"Topic counts saved to {topic_counts_file}")

    df = df.dropna(subset=['Topics'])
    return df, topic_counts


def apply_tsne_and_cluster(df, n_clusters, topic_embeddings, unique_topics, output_directory, perplexities):
    logging.info(f"Applying clustering with {n_clusters} clusters...")

    # Cluster the topics using the specified number of clusters
    topic_cluster_map = cluster_topics(topic_embeddings, n_clusters, unique_topics)

    topic_names = list(topic_embeddings.keys())
    embeddings_list = list(topic_embeddings.values())
    embeddings_array = np.array(embeddings_list)

    color_list = [
        '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
        '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
        '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000',
        '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080'
    ]
    custom_cmap = ListedColormap(color_list)

    # Apply the clustering results to the DataFrame
    def map_topics_to_clusters(topics_str):
        topics = [topic.strip() for topic in topics_str.split(',')]
        clusters = [topic_cluster_map.get(topic, -1) for topic in topics]  # -1 if topic not found
        return clusters

    df['Topic_Clusters'] = df['Topics'].apply(map_topics_to_clusters)

    # Save the DataFrame with clusters to a file
    output_file_path_df = os.path.join(output_directory, f'14_final_dataframe_with_topic_clusters_{n_clusters}.csv')
    df.to_csv(output_file_path_df, index=False)
    logging.info(f"DataFrame with {n_clusters} topic clusters saved to {output_file_path_df}")

    # Save the clusters to a file with a dynamic name
    cluster_file_name = os.path.join(output_directory, f'15_topics_grouped_by_clusters_{n_clusters}.txt')
    clustered_topics = print_topic_clusters(topic_cluster_map, cluster_file_name)

    logging.info(f"Clusters saved to {cluster_file_name}")

    # Loop over the given perplexities
    for perplexity in perplexities:
        logging.info(f"Applying t-SNE with perplexity {perplexity}...")

        # Apply t-SNE to reduce dimensionality to 2 components (2D)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embeddings_2d = tsne.fit_transform(embeddings_array)

        # Create a list of cluster labels for each topic
        cluster_labels = [topic_cluster_map.get(topic, -1) for topic in unique_topics]

        # Plot the 2D t-SNE visualization
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=cluster_labels, cmap=custom_cmap, s=50,
                              edgecolor='k')
        plt.title(f't-SNE visualization of {n_clusters} Topic Clusters with perplexity {perplexity}')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')

        # Explicitly create the legend using unique cluster labels
        unique_labels = np.unique(cluster_labels)
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_list[i], markersize=10) for i in
                   unique_labels]
        plt.legend(handles, unique_labels, title="Clusters", loc="best")

        # Save the plot as an image file (e.g., PNG or JPEG)
        plot_save_path = os.path.join(output_directory, f'16_tsne_topic_clusters_{n_clusters}_perplexity_{perplexity}.png')
        plt.savefig(plot_save_path, format='png', dpi=300)  # Save with high resolution
        logging.info(f"t-SNE plot for {n_clusters} clusters with perplexity {perplexity} saved to {plot_save_path}")

        plt.grid(True)
        plt.show()


def get_topic_embeddings(unique_topics, output_directory):
    embeddings_file_path = os.path.join(output_directory, '13_topic_embeddings.json')
    if config.reuse["embeddings"]:
        if not os.path.exists(embeddings_file_path):
            logging.error(f"Embeddings file {embeddings_file_path} not found.")
            return None
        with open(embeddings_file_path, 'r') as f:
            topic_embeddings = json.load(f)
            logging.info(f"Topic embeddings loaded from {embeddings_file_path}")
            return topic_embeddings
    else:
        topic_embeddings = {}
        for topic in unique_topics:
            embedding = get_embedding(topic, model='text-embedding-3-small')
            topic_embeddings[topic] = embedding

        with open(embeddings_file_path, 'w') as f:
            json.dump({topic: embedding for topic, embedding in topic_embeddings.items()}, f)
        logging.info(f"Topic embeddings saved to {embeddings_file_path}")
        return topic_embeddings


def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def get_unique_topics(topic_counts_file):
    topic_counts = {}
    with open(topic_counts_file, 'r') as f:
        for line in f:
            topic, count = line.strip().split(': ')
            topic_counts[topic] = int(count)

    # Step 1: Extract Unique Topics
    unique_topics = list(topic_counts.keys())
    logging.info(f"Number of unique topics: {len(unique_topics)}")
    return unique_topics


def build_embeddings_for_topics(unique_topics, output_directory):
    topic_embeddings = {}
    for topic in unique_topics:
        embedding = get_embedding(topic, model='text-embedding-3-small')
        topic_embeddings[topic] = embedding

    embeddings_file_path = os.path.join(output_directory, 'topic_embeddings.json')
    with open(embeddings_file_path, 'w') as f:
        json.dump({topic: embedding for topic, embedding in topic_embeddings.items()}, f)
    logging.info(f"Topic embeddings saved to {embeddings_file_path}")
    return topic_embeddings


def read_embeddings_from_json(output_directory):
    embeddings_file_path = os.path.join(output_directory, 'topic_embeddings.json')

    if not os.path.exists(embeddings_file_path):
        logging.error(f"Embeddings file {embeddings_file_path} not found.")
        return None

    with open(embeddings_file_path, 'r') as f:
        topic_embeddings = json.load(f)

    logging.info(f"Topic embeddings loaded from {embeddings_file_path}")
    return topic_embeddings


def cluster_topics(embeddings, n_clusters, unique_topics):
    embedding_matrix = np.array(list(embeddings.values()))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    topic_labels = kmeans.fit_predict(embedding_matrix)

    # Map each topic to its cluster label
    topic_cluster_map = dict(zip(unique_topics, topic_labels))
    return topic_cluster_map





def print_topic_clusters(topic_cluster_map, output_file_path_clusters):
    clustered_topics = {}
    for topic, cluster_label in topic_cluster_map.items():
        if cluster_label not in clustered_topics:
            clustered_topics[cluster_label] = []
        clustered_topics[cluster_label].append(topic)

    # Save the topics grouped by clusters
    with open(output_file_path_clusters, 'w') as f:
        for cluster, topics in clustered_topics.items():
            f.write(f"Cluster {cluster}:\n")
            f.write(', '.join(topics))
            f.write("\n\n")
    logging.info(f"Topics grouped by clusters saved to {output_file_path_clusters}")
    return clustered_topics
