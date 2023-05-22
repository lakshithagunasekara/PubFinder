import subprocess
import config
import pandas as pd
import os


def extract_search_results(source_name):
    temp_start_year = config.start_year
    temp_end_year = config.start_year + 1
    while temp_end_year <= config.end_year:
        path = 'sources/' + source_name + '_' + str(temp_end_year) + '.csv'
        if os.path.isfile(path):
            delete_command = 'rm ' + path
            subprocess.run(delete_command, shell=True)

        pop_command = 'pop8query --"' + source_name + '" --keywords "' + config.keyword + '" --years=' \
                      + str(temp_start_year) + "-" + str(temp_end_year) + ' --format CSV >> ' + path
        print(pop_command, '\n')
        temp_start_year = temp_end_year
        temp_end_year += 1
        subprocess.run(pop_command, shell=True)


if not config.only_filter:
    for source in config.sources:
        print("Exporting the results for " + source)
        extract_search_results(source)

df_append = pd.DataFrame()

print("Append all the sources together")
for file in os.listdir('sources'):
    file_location = 'sources/' + file
    if os.stat(file_location).st_size != 0 :
        df_temp = pd.read_csv(file_location)
        df_append = df_append.append(df_temp, ignore_index=True)

df_append.to_csv('results/combined.csv', index=False)
print("Appended file saved to results/combined.csv")

if config.consider_year_for_filter:
    duplicate_filter = ['Title', 'Year']
else:
    duplicate_filter = ['Title']

print("Filter duplicates")
df_append.sort_values("Title", inplace=True)
df_append.drop_duplicates(subset=duplicate_filter, keep='first', inplace=True)

df_append['Cites'] = df_append['Cites'].astype(int)
df_append.sort_values("Cites", inplace=True, ascending=False)
df_append.to_csv('results/duplicates_removed.csv', index=False)
print("Duplicate remove file saved to results/duplicates_removed.csv")


df_filtered = pd.DataFrame()

for filter in config.title_filters:
    print("Applying filter '"+ filter + "' to the title")
    new_df = df_append[df_append['Title'].str.contains(filter, na=False, case=False)]
    new_df.to_csv('results/filtered_' + filter + '.csv', index=False)
    df_filtered = df_filtered.append(new_df, ignore_index=True)

df_filtered.sort_values("Title", inplace=True)
df_filtered.drop_duplicates(subset=duplicate_filter, keep='first', inplace=True)

df_filtered['Cites'] = df_filtered['Cites'].astype(int)
df_filtered.sort_values("Cites", inplace=True, ascending=False)
df_filtered.to_csv('results/filtered.csv', index=False)
print("Filtered file saved to results/filtered.csv")


