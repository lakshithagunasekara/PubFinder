import json
import os

import requests
import pandas as pd
import logging as log
import config


def get_publication_metadate(doi, current_api_key):
    url = "https://api.elsevier.com/content/abstract/doi/" + doi + "?apiKey=" + current_api_key + "&httpAccept=application%2Fjson"
    response = requests.get(url)
    return response


def fetch_abstract_from_elsevier(doi):
    api_keys = config.elsevier_api_keys
    api_key_for_metadata_request = api_keys[0]
    metadata_response = get_publication_metadate(doi, api_key_for_metadata_request)
    if metadata_response.status_code == 429:
        log.error("Meta data response " + str(metadata_response.status_code))
        rotations = 0
        while rotations < len(config.elsevier_api_keys) and metadata_response.status_code == 429:
            rotations += 1
            api_key_for_metadata_request = api_keys[rotations]
            metadata_response = get_publication_metadate(doi, api_key_for_metadata_request)

    if metadata_response.status_code == 200:
        metadata_response_json = json.loads(metadata_response.content)
        if metadata_response_json.get('abstracts-retrieval-response') is not None:
            try:
                abstract = metadata_response_json["abstracts-retrieval-response"]["item"]["bibrecord"]["head"][
                    "abstracts"]
                return abstract, metadata_response.status_code, metadata_response_json  # Return abstract and response details
            except Exception as e:
                log.error(f"Error extracting abstract for {doi}: {e}")

    return None, metadata_response.status_code, metadata_response.content  # Return None for abstract but include status code and response content


def fetch_abstract_from_semantic_scholar(doi):
    """
    Fetches the abstract of a publication from Semantic Scholar using its DOI.
    """
    url = f"https://api.semanticscholar.org/v1/paper/{doi}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            paper_info = response.json()
            return paper_info.get('abstract',
                                  None), response.status_code, paper_info  # Return abstract and response details
        else:
            log.error(f"Failed to fetch from Semantic Scholar for DOI {doi}: Status Code {response.status_code}")
            return None, response.status_code, response.content  # Return None for abstract but include status code and response content
    except Exception as e:
        log.error(f"Error fetching from Semantic Scholar for DOI {doi}: {e}")
        return None, None, str(e)  # Return None for abstract and status code, include error message


def populate_missing_abstracts(df, output_directory):
    iteration = 0
    api_responses = []
    log.info("Populating missing abstracts")

    api_response_dict = {}
    response_file_location = os.path.join(output_directory, "api_responses.csv")
    log.info(f"Loading existing API responses from {response_file_location}")

    # Check if the API response file exists and load it
    if os.path.isfile(response_file_location) and os.stat(response_file_location).st_size != 0:
        api_response_df = pd.read_csv(response_file_location)
        api_response_dict = api_response_df.set_index('DOI')['Abstract'].to_dict()
    else:
        log.warning(f"API response file {response_file_location} not found or empty. Proceeding with API calls.")

    # Process each row to populate missing abstracts
    for index, row in df.iterrows():
        iteration += 1
        doi = row.get('DOI', None)
        log.info(f"Processing {iteration} DOI: {doi}")

        # Check if the abstract is missing
        if pd.isna(row['Abstract']) or not str(row['Abstract']).strip() or row['Abstract'] == 0:
            if doi:
                # Use cached abstract if available
                abstract = api_response_dict.get(doi)
                if abstract:
                    log.info(f"Abstract found in cached responses for DOI {doi}")

                # Fetch abstract from external API if not cached
                if not abstract:
                    abstract, status_code, response_content = fetch_abstract_from_elsevier(doi)

                    # Record the API response
                    api_responses.append({
                        'DOI': doi,
                        'Status_Code': status_code,
                        'Response': json.dumps(response_content) if isinstance(response_content, (dict, list)) else str(
                            response_content),
                        'Abstract': abstract if abstract else "None"
                    })

                # Update the DataFrame if an abstract is found
                if abstract:
                    df.at[index, 'Abstract'] = abstract
                    log.info(f"Abstract populated for DOI {doi}")

        # Save API responses every 100 iterations to prevent data loss on interruption
        if iteration % 100 == 0 and api_responses:
            log.info(f"Saving API responses at iteration {iteration}")
            save_api_responses(api_responses, response_file_location)
            api_responses = []  # Clear the list after saving to file

    # Final save of any remaining API responses
    if api_responses:
        save_api_responses(api_responses, response_file_location)

    # Save the updated DataFrame with populated abstracts
    df.to_csv(f'{output_directory}/6_populate_abstracts.csv', index=False)
    log.info(f"Abstracts populated file is saved to {output_directory}/6_populate_abstracts.csv")

    return df


def save_api_responses(api_responses, response_file_location):
    """Helper function to save or append API responses to a CSV file."""
    new_responses_df = pd.DataFrame(api_responses)

    # Check if the file exists
    if os.path.isfile(response_file_location) and os.stat(response_file_location).st_size != 0:
        # If the file exists, load the existing responses and append new ones
        existing_responses_df = pd.read_csv(response_file_location)
        combined_df = pd.concat([existing_responses_df, new_responses_df], ignore_index=True).drop_duplicates(
            subset=['DOI'])
        combined_df.to_csv(response_file_location, index=False)
        log.info(f"Appended and saved new API responses to {response_file_location}")
    else:
        # If the file doesn't exist, create it with the new responses
        new_responses_df.to_csv(response_file_location, index=False)
        log.info(f"Saved new API responses to {response_file_location}")

