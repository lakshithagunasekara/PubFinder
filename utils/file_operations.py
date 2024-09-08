import os
import shutil
import zipfile
from datetime import datetime
import logging as log

import config


def prepare_directories():
    output_directory = 'results'
    sources_directory = 'sources'
    archive_directory = 'archive'

    folders_to_clear = []
    if not config.reuse["sources"]:
        folders_to_clear.append(sources_directory)

    if not (config.reuse["abstract_responses"] or config.reuse["topics"] or config.reuse["embeddings"]):
        folders_to_clear.append(output_directory)

    for directory in folders_to_clear:
        if os.path.exists(directory):
            log.info(f"Clearing the directory {directory}")
            shutil.rmtree(directory)  # Remove the directory and its contents

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    if not os.path.exists(sources_directory):
        os.makedirs(sources_directory)
    if not os.path.exists(archive_directory):
        os.makedirs(archive_directory)

    return output_directory, sources_directory, archive_directory


def zip_results(sources_directory, results_directory, output_directory, config_file_path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = os.path.join(output_directory, f'results_{timestamp}.zip')

    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for foldername, subfolders, filenames in os.walk(sources_directory):
            for filename in filenames:
                file_path = os.path.join(foldername, filename)
                zipf.write(file_path, os.path.join('sources', os.path.relpath(file_path, sources_directory)))

        for foldername, subfolders, filenames in os.walk(results_directory):
            for filename in filenames:
                file_path = os.path.join(foldername, filename)
                zipf.write(file_path, os.path.join('results', os.path.relpath(file_path, results_directory)))

        zipf.write(config_file_path, os.path.join('config', os.path.basename(config_file_path)))
        zipf.write("output.log", os.path.join('logs', "output.log"))
    log.info(f"Results, sources, and config zipped into {zip_filename}")


def save_config_to_text_file(config_module, output_directory):
    config_file_path = os.path.join(output_directory, "config.txt")
    with open(config_file_path, 'w') as file:
        for key in dir(config_module):
            if not key.startswith('__'):
                value = getattr(config_module, key)
                file.write(f'{key} = {value}\n')
    return config_file_path
