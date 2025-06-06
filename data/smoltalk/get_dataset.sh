#!/bin/bash

### Instructions:
# 1. Replace "INSERT_URL_WITH_FILES" with the actual URL to the Parquet files.
# 2. Modify the "include_keys" array to specify the keys you want to include in the output.
# 3. (Optionally) Modify the "value_prefixes" array to set prefixes for each value, use "" for empty prefixes
# 4. Set "--skip_empty" to true if you want to skip empty fields, or false if not needed.
# 5. Set "--no_output_text" to true if you plan to process the intermediate json files in a custom manner.

# Run the Python script with the specified arguments

# Add url with dataset here:
url="https://huggingface.co/datasets/HuggingFaceTB/smoltalk/tree/main/data/smol-summarize"

python3 ./utils/get_nested_parquet_dataset.py \
    --url "${url}" \
    -o output.txt \
    -i content \
    --list_key messages \
    --role_prefixes user $'#U:\n' assistant $'#B:\n' \
    -s

