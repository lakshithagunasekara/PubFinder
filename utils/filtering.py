import pandas as pd

import config
import logging as log
from openai import OpenAI

client = OpenAI(
    api_key="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")


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

    return filtered_by_cites


def filter_titles_by_mandatory_keywords(df, output_directory):
    log.info("Filtering titles based on keywords")
    for filter in config.mandatory_filters:
        keywords = filter.split(",")
        df_mandatory = pd.DataFrame()
        for keyword in keywords:
            df_mandatory_filtered_for_keyword_title = df[df['Title'].str.contains(keyword, na=False, case=False)]
            df_mandatory = pd.concat([df_mandatory, df_mandatory_filtered_for_keyword_title], ignore_index=True)
        df = df_mandatory
        df_mandatory.to_csv(f'{output_directory}/4_filtered_title_{filter}.csv', index=False)

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

    for filter in config.mandatory_filters:
        keywords = filter.split(",")
        df_mandatory = pd.DataFrame()
        for keyword in keywords:
            df_mandatory_filtered_abstract = df[df['Abstract'].str.contains(keyword, na=False, case=False)]
            df_mandatory = pd.concat([df_mandatory, df_mandatory_filtered_abstract], ignore_index=True)
        df = df_mandatory
        df_mandatory_filtered_abstract.to_csv(f'{output_directory}/7_filtered_abstract_{keyword}.csv', index=False)

    df_mandatory.sort_values("Title", inplace=True)
    df_mandatory.drop_duplicates(subset=['Title', 'Year'], keep='first', inplace=True)
    df_mandatory.drop_duplicates(subset=['DOI'], keep='first', inplace=True)

    df_mandatory['Cites'] = df_mandatory['Cites'].astype(int)
    df_mandatory.sort_values("Cites", inplace=True, ascending=False)

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

    df.to_csv(f'{output_directory}/9_filtered_optional_keywords.csv', index=False)
    log.info(f"Optional Filtered file saved to {output_directory}/9_filtered_optional_keywords.csv")
    return df


def create_prompt(entries, mandatory_filters, optional_filters):
    """
    Creates a prompt for the LLM to determine relevance and tag entries based on filters.

    Args:
    entries (list): A list of tuples containing (Title, Abstract).
    mandatory_filters (list): A list of mandatory keywords to check relevance against.
    optional_filters (list): A list of optional filters to identify specific discussions.

    Returns:
    str: A formatted prompt string for the LLM.
    """
    prompt = (
        "You are a highly intelligent AI that determines if scientific papers are relevant "
        "to a set of mandatory keywords and tags them with optional filters if applicable. "
        "Given the following titles and abstracts, determine if each one is relevant based on "
        "the mandatory keywords: {}. Additionally, identify which of the following optional "
        "filters it discusses: {}.\n\n"
    ).format(", ".join(mandatory_filters), ", ".join(optional_filters))

    prompt += (
        "For each title and abstract, answer 'Relevant' or 'Irrelevant' based on the mandatory "
        "keywords. If an abstract is relevant but does not discuss any optional filters, "
        "consider it 'Irrelevant'. List any applicable optional filters next to 'Relevant'.\n"
        "Format your response as '1. Relevant, principles, strategies' or '2. Irrelevant'.\n\n"
    )

    for i, (title, abstract) in enumerate(entries, start=1):
        prompt += f"Title {i}: {title}\nAbstract {i}: {abstract}\n\n"

    prompt += "Provide your answers in the specified format."
    return prompt


def filter_keywords_with_gpt(df, mandatory_filters, optional_filters, output_directory, batch_size=5):
    """
    Processes batches of entries using GPT-4 Turbo to determine relevance and optional tags.

    Args:
    df (pd.DataFrame): DataFrame containing the entries to process.
    mandatory_filters (list): A list of mandatory keywords to check relevance against.
    optional_filters (list): A list of optional filters to identify specific discussions.
    batch_size (int): Number of entries per batch.

    Returns:
    pd.DataFrame: Updated DataFrame with relevance and tags determined by GPT-4 Turbo.
    """

    # Prepare DataFrame to store results
    df['Relevance'] = None
    df['Tags'] = None

    # Iterate over DataFrame in batches
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i + batch_size]
        entries = list(zip(batch['Title'], batch['Abstract']))
        prompt = create_prompt(entries, mandatory_filters, optional_filters)

        try:
            response = client.chat.completions.create(model="gpt-4o-2024-08-06",
                                                      messages=[
                                                          {"role": "system", "content": "You are a helpful assistant."},
                                                          {"role": "user", "content": prompt}
                                                      ],
                                                      max_tokens=500)

            # Extract and parse the LLM response
            results = response.choices[0].message.content.splitlines()

            # Update DataFrame with LLM's relevance judgments and tags
            for j, result in enumerate(results):
                relevance, tags = result.split(',', 1) if ',' in result else (result, '')
                relevance = relevance.strip().split('.')[1].strip()  # Extract "Relevant" or "Irrelevant"
                tags = tags.strip()

                # If no tags are provided and the response is 'Relevant', mark it as 'Irrelevant'
                if relevance == "Relevant" and not tags:
                    relevance = "Irrelevant"

                df.at[batch.index[j], 'Relevance'] = relevance
                df.at[batch.index[j], 'Tags'] = tags if tags else "None"
                log.info(f"Processed entry {batch.index[j]}: {relevance}, Tags: {tags}")

        except Exception as e:
            log.error(f"Error processing batch starting at index {i}: {e}")
            continue

    df = df[df['Relevance'] == 'Relevant']
    df.to_csv(f'{output_directory}/10_filtered_keywords_with_gpt.csv', index=False)
    log.info(f"Keywords filtered with GPT saved to {output_directory}/10_filtered_keywords_with_gpt.csv")
    return df
