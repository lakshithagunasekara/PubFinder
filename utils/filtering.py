import pandas as pd

import config
import logging as log
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List

from utils.topics_utilts import preprocess_topic

client = OpenAI(api_key=config.openai_key)


class JournalFilterResponse(BaseModel):
    journal_id: int
    relevant: bool
    tags: List[str]


class GPTFilterResponse(BaseModel):
    journals: List[JournalFilterResponse]


def filter_duplicates(df, output_directory):
    log.info("Filtering duplicates")

    valid_abstract_mask = df['Abstract'].notna() & df['Abstract'].str.strip().ne('')
    df['HasValidAbstract'] = valid_abstract_mask.astype(int)
    df.sort_values(by='HasValidAbstract', ascending=False, inplace=True)

    duplicate_filter = ['Title', 'Year'] if config.consider_year_for_filter else ['Title']
    df.drop_duplicates(subset=duplicate_filter, keep='first', inplace=True)
    df.drop_duplicates(subset=['DOI'], keep='first', inplace=True)

    df['Cites'] = df['Cites'].astype(int)
    df.sort_values("Cites", inplace=True, ascending=False)
    df = df.reset_index(drop=True)

    df.to_csv(f'{output_directory}/2_filtered_duplicates_removed.csv', index=False)
    log.info(f"Duplicate removed file saved to {output_directory}/2_filtered_duplicates_removed.csv")
    return df


def filter_by_cites(df, output_directory):
    log.info("Filtering based on citations")
    last_year = config.end_year
    df = df.fillna(0)
    df['Cites'] = df['Cites'].astype(int)
    df['Year'] = df['Year'].astype(int)
    df.sort_values("Year", inplace=True, ascending=False)

    last_year_df = df[df['Year'] == last_year]
    last_year_df.sort_values("Cites", inplace=True, ascending=False)
    filtered_by_cites_last_year = last_year_df[last_year_df["Cites"] >= config.most_recent_year_citation]

    year_older_df = df[df['Year'] != last_year]
    year_older_df.sort_values("Cites", inplace=True, ascending=False)
    filtered_by_cites_old = year_older_df[year_older_df["Cites"] >= config.min_citations]

    filtered_by_cites = pd.concat([filtered_by_cites_old, filtered_by_cites_last_year], ignore_index=True)

    filtered_by_cites.to_csv(f'{output_directory}/3_filtered_by_citations.csv', index=False)
    log.info(f"filtered_by_cites file saved to {output_directory}/3_filtered_by_citations.csv")
    filtered_by_cites = filtered_by_cites.reset_index(drop=True)
    return filtered_by_cites


def filter_titles_by_mandatory_keywords(df, output_directory):
    log.info("Filtering titles based on keywords")
    counter = 0
    for filter in config.mandatory_filters:
        counter += 1
        keywords = filter.split(",")
        df_mandatory = pd.DataFrame()
        for keyword in keywords:
            df_mandatory_filtered_for_keyword_title = df[df['Title'].str.contains(keyword, na=False, case=False)]
            df_mandatory = pd.concat([df_mandatory, df_mandatory_filtered_for_keyword_title], ignore_index=True)
        df = df_mandatory
        df_mandatory.to_csv(f'{output_directory}/4_filtered_title_{counter}.csv', index=False)

    df_mandatory.sort_values("Title", inplace=True)
    df_mandatory.drop_duplicates(subset=['Title', 'Year'], keep='first', inplace=True)
    df_mandatory.drop_duplicates(subset=['DOI'], keep='first', inplace=True)

    df_mandatory['Cites'] = df_mandatory['Cites'].astype(int)
    df_mandatory.sort_values("Cites", inplace=True, ascending=False)
    df_mandatory = df_mandatory.reset_index(drop=True)

    df_mandatory.to_csv(f'{output_directory}/5_filtered_title_mandatory_keywords.csv', index=False)
    log.info(f"Mandatory filtered file saved to {output_directory}/5_filtered_title_mandatory_keywords.csv")
    return df_mandatory


def filter_abstract_by_mandatory_keywords(df, output_directory):
    log.info("Filtering based on keywords")
    counter = 0
    for filter in config.mandatory_filters:
        counter += 1
        keywords = filter.split(",")
        df_mandatory = pd.DataFrame()
        for keyword in keywords:
            processed_keyword = preprocess_topic(keyword)
            df_mandatory_filtered_abstract = df[df['Processed_Abstract'].str.contains(processed_keyword, na=False, case=False)]
            df_mandatory = pd.concat([df_mandatory, df_mandatory_filtered_abstract], ignore_index=True)
        df = df_mandatory
        df_mandatory_filtered_abstract.to_csv(f'{output_directory}/7_filtered_abstract_{counter}.csv', index=False)

    df_mandatory.sort_values("Title", inplace=True)
    df_mandatory.drop_duplicates(subset=['Title', 'Year'], keep='first', inplace=True)
    df_mandatory.drop_duplicates(subset=['DOI'], keep='first', inplace=True)

    df_mandatory['Cites'] = df_mandatory['Cites'].astype(int)
    df_mandatory.sort_values("Cites", inplace=True, ascending=False)
    df_mandatory = df_mandatory.reset_index(drop=True)

    df_mandatory.to_csv(f'{output_directory}/8_filtered_abstract_mandatory_keywords.csv', index=False)
    log.info(f"Mandatory filtered file saved to {output_directory}/8_filtered_abstract_mandatory_keywords.csv")
    return df_mandatory


def filter_by_optional_keywords(input_df, output_directory):
    for filter in config.optional_filter:
        log.info(f"Applying filter '{filter}' to the title")
        title_df = input_df[input_df['Title'].str.contains(filter, na=False, case=False)]
        abstract_df = input_df[input_df['Abstract'].str.contains(filter, na=False, case=False)]

        title_df = pd.concat([title_df, abstract_df], ignore_index=True)
        title_df.sort_values("Title", inplace=True)
        title_df.drop_duplicates(subset=['Title', 'Year'], keep='first', inplace=True)
        title_df.drop_duplicates(subset=['DOI'], keep='first', inplace=True)

        title_df.to_csv(f'{output_directory}/6_optional_filtered_{filter}.csv', index=False)
        df = pd.concat([df, title_df], ignore_index=True)

    df.sort_values("Title", inplace=True)
    df.drop_duplicates(subset=['Title', 'Year'], keep='first', inplace=True)
    df.drop_duplicates(subset=['DOI'], keep='first', inplace=True)

    df['Cites'] = df['Cites'].astype(int)
    df.sort_values("Cites", inplace=True, ascending=False)
    df = df.reset_index(drop=True)
    df.to_csv(f'{output_directory}/9_filtered_optional_keywords.csv', index=False)
    log.info(f"Optional Filtered file saved to {output_directory}/9_filtered_optional_keywords.csv")
    return df


def classify_relevancy_gpt_batch(journals_batch, mandatory_filters, optional_filters):
    prompt = (
        f"Please read the following title and abstract pairs, determine if each one is relevant based on "
        "the mandatory keywords: {}. Additionally, identify which of the following optional "
        "filters it discusses: {}.\n\n"
        "For each title and abstract, answer 'Relevant' or 'Irrelevant' based on the mandatory "
        "keywords. List any applicable optional filters as well'.\n"
        f"Return the result as a structured JSON format with 'journal_id', 'relevant' and 'tags'.\n\n"
    ).format(", ".join(mandatory_filters), ", ".join(optional_filters))

    for i, journal in enumerate(journals_batch, start=1):
        title = journal['Title']
        abstract = journal['Abstract']
        prompt += f"Journal {i}:\nTitle: {title}\nAbstract: {abstract}\n\n"

    response = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system",
             "content": "You are a highly intelligent AI that determines if scientific papers are relevant "
                        "to a set of mandatory keywords and tags them with optional filters if applicable."},
            {"role": "user", "content": prompt}
        ],
        response_format=GPTFilterResponse
    )

    return response.choices[0].message.parsed


def filter_articles_with_gpt(df, output_directory, mandatory_filters, optional_filters):
    batch_size = 5
    df["Relevance"] = ""
    df["Tags"] = ""

    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i + batch_size]

        journals_batch = [
            {"Title": row['Title'], "Abstract": row['Abstract']}
            for _, row in batch_df.iterrows()
        ]

        gpt_response = classify_relevancy_gpt_batch(journals_batch, mandatory_filters, optional_filters)

        for journal_response in gpt_response.journals:
            journal_id = i + journal_response.journal_id - 1
            relevance = journal_response.relevant
            tags = [tag for tag in journal_response.tags]
            if relevance == True and len(tags) == 0:
                relevance = False
            df.at[journal_id, 'Relevance'] = relevance
            df.at[journal_id, 'Tags'] = tags if tags else "None"
            log.info(f"Processed entry {journal_id}: {relevance}, Tags: {tags}")

    df = df[df['Relevance'] == True]
    df = df.reset_index(drop=True)
    df.to_csv(f'{output_directory}/10_filtered_keywords_with_gpt.csv', index=False)
    log.info(f"Keywords filtered with GPT saved to {output_directory}/10_filtered_keywords_with_gpt.csv")
    return df
