import json
import os

import pandas as pd

from utils.extract_results import load_search_results
from utils.file_operations import zip_results, save_config_to_text_file, prepare_directories
from utils.filtering import filter_duplicates, filter_by_cites, \
     filter_by_optional_keywords, filter_abstract_by_mandatory_keywords, \
    filter_titles_by_mandatory_keywords, filter_articles_with_gpt
import config
from utils.pub_utils import populate_missing_abstracts
import logging as log
from utils.topics_utilts import preprocess_text, get_topics_for_journals_df, apply_tsne_and_cluster, \
    get_topic_embeddings

log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    log.FileHandler("output.log"),  # Save log messages to 'output.log'
                    log.StreamHandler()  # Also output log messages to the console
                ])

row_counts = {}
log.info("Starting the script")
log.info("Configuration settings: \n" + json.dumps({
    "keywords": config.keywords,
    "mandatory_filters": config.mandatory_filters,
    "start_year": config.start_year,
    "end_year": config.end_year,
    "sources": config.sources,
    "only_cluster": config.only_cluster,
    "reuse": config.reuse
}, indent=4))

journal_data_df = pd.DataFrame

output_directory, sources_directory, archive_directory = prepare_directories()

if not config.only_cluster:
    journal_data_df = load_search_results(config.sources, sources_directory, output_directory)
    row_counts['Loaded'] = len(journal_data_df)
    log.info(f"Rows loaded {len(journal_data_df)}")

    # Remove duplicates
    journal_data_df = filter_duplicates(journal_data_df, output_directory)
    row_counts['After Removing Duplicates'] = len(journal_data_df)
    log.info(f"After Removing Duplicates {len(journal_data_df)}")

    if config.filter_by_citations:
        journal_data_df = filter_by_cites(journal_data_df, output_directory)
        row_counts['After Filtering by Citations'] = len(journal_data_df)
        log.info(f"After Filtering by Citations {len(journal_data_df)}")

    if config.allow_filter_by_mandatory_keywords:
        #journal_data_df = filter_titles_by_mandatory_keywords(journal_data_df, output_directory)
        row_counts['After Filtering Titles by Mandatory Keywords'] = len(journal_data_df)
        log.info(f"After Filtering Titles by Mandatory Keywords {len(journal_data_df)}")

    # Populate missing abstracts
    journal_data_df = populate_missing_abstracts(journal_data_df, output_directory)
    row_counts['After Populating Abstracts'] = len(journal_data_df)
    log.info(f"After Populating Abstracts {len(journal_data_df)}")

    # Drop entries without abstracts and titles
    journal_data_df = journal_data_df[journal_data_df['Abstract'].apply(lambda x: isinstance(x, str) and x != '0')]
    journal_data_df = journal_data_df[journal_data_df['Title'].apply(lambda x: isinstance(x, str))]
    journal_data_df = journal_data_df.reset_index(drop=True)
    row_counts['After Dropping Entries Without Abstracts or Titles'] = len(journal_data_df)
    log.info(f"After Dropping Entries Without Abstracts or Titles {len(journal_data_df)}")

    # Apply preprocessing to the 'Abstract' and 'Title' columns
    journal_data_df['Processed_Abstract'] = journal_data_df['Abstract'].apply(preprocess_text)
    journal_data_df['Processed_Title'] = journal_data_df['Title'].apply(preprocess_text)

    if config.allow_filter_by_mandatory_keywords:
        journal_data_df = filter_abstract_by_mandatory_keywords(journal_data_df, output_directory)
        row_counts['After Filtering Abstracts by Mandatory Keywords'] = len(journal_data_df)
        log.info(f"After Filtering Abstracts by Mandatory Keywords {len(journal_data_df)}")

    # Filter by Keywords
    if config.allow_filter_by_optional_keywords:
        # journal_data_df = filter_by_optional_keywords(journal_data_df, output_directory)

        journal_data_df = filter_articles_with_gpt(journal_data_df, output_directory, config.mandatory_filters,
                                                   config.optional_filter)
        row_counts['After Filtering by Optional Keywords'] = len(journal_data_df)
        log.info(f"After Filtering by Optional Keywords {len(journal_data_df)}")

    journal_data_df = journal_data_df.reset_index(drop=True)
    row_counts['Before processing'] = len(journal_data_df)
    log.info(f"Before processing {len(journal_data_df)}")


    journal_data_df['Topics'] = ""

journal_data_df, topic_counts = get_topics_for_journals_df(journal_data_df, output_directory)
row_counts['After assigning topics'] = len(journal_data_df)
log.info(f"After assigning topics {len(journal_data_df)}")

unique_topics = list(topic_counts.keys())
log.info(f"Unique Topics counts : {len(unique_topics)}")

log.info("Generating GPT embeddings for unique topics...")
topic_embeddings = get_topic_embeddings(unique_topics, output_directory)

cluster_limits = [4, 5, 6, 7, 8, 9, 10]
perplexities = [5, 10, 20, 30, 40]
for n_clusters in cluster_limits:
    apply_tsne_and_cluster(journal_data_df, n_clusters, topic_embeddings, unique_topics, output_directory, perplexities)

# Save config and zip results
config_file_path = save_config_to_text_file(config, output_directory)

row_counts_file = os.path.join(output_directory, "row_counts.txt")
with open(row_counts_file, "w") as f:
    for stage, count in row_counts.items():
        f.write(f"{stage}: {count}\n")

zip_results(sources_directory, output_directory, archive_directory, config_file_path)

# Remove the config file after zipping
os.remove(config_file_path)
