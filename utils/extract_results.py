import os
import subprocess
import pandas as pd
import config
import logging as log


def load_search_results(sources, sources_directory, output_directory):
    journal_data_df = pd.DataFrame()
    loaded_files_location = os.path.join(output_directory, '1_combined_initial_search_results.csv')
    if config.reuse["sources"]:
        journal_data_df = pd.read_csv(loaded_files_location)
        log.info(f"Loaded the file from {loaded_files_location}")
    else:
        for source in sources:
            log.info(f"Exporting the results for {source}")
            try:
                extract_search_results_for_source(source, config.start_year, config.end_year, config.keywords)
            except Exception as e:
                log.error(f"Error extracting results for {source}: {e}")

        log.info("Append all the sources together")
        for file in os.listdir(sources_directory):
            if file.endswith(".csv"):
                file_location = os.path.join(sources_directory, file)
                log.info(f"Processing file: {file_location}")
                if os.stat(file_location).st_size != 0:
                    df_temp = pd.read_csv(file_location)
                    journal_data_df = pd.concat([journal_data_df, df_temp], ignore_index=True)
            else:
                log.info(f"Skipping non-CSV file: {file}")

        journal_data_df.to_csv(loaded_files_location, index=False)
        log.info(f"Appended file saved to {loaded_files_location}")
    return journal_data_df


def extract_search_results_for_source(source_name, start_year, end_year, keywords):
    temp_start_year = start_year
    temp_end_year = start_year + 1
    while temp_start_year <= end_year:
        for keyword in keywords:
            keyword_without_space = keyword.replace(" ", "_")
            path = f'sources/{keyword_without_space}_{source_name}_{temp_end_year}.csv'
            if os.path.isfile(path):
                delete_command = f'rm {path}'
                subprocess.run(delete_command, shell=True)

            pop_command = f'pop8query --"{source_name}" --keywords "{keyword}" --years={temp_start_year}-{temp_end_year} --format CSV >> {path}'
            log.info(f"{pop_command}\n")
            log.info(f"Results will be saved in {path}")
            subprocess.run(pop_command, shell=True)
        temp_start_year = temp_end_year
        temp_end_year += 1
