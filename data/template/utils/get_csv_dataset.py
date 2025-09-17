import argparse
import csv
import os
import re
from typing import Dict, Iterable, List, Optional, TextIO

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


class OutputManager:
    """Helper class to handle combined and per-prefix output writing."""

    def __init__(self, combined_file, prefix_files):
        self.combined_file = combined_file
        self.combined_written = False if combined_file is not None else None
        self.prefix_files = prefix_files or {}
        self.prefix_written: Dict[str, bool] = {
            prefix: False for prefix in self.prefix_files
        }

    def write(self, prefix: str, value: str) -> None:
        text_value = "" if value is None else str(value)
        combined_line = f"{prefix}{text_value}"
        trimmed_combined = combined_line.strip()

        if self.combined_file is not None:
            if self.combined_written:
                self.combined_file.write("\n")
            self.combined_file.write(trimmed_combined)
            self.combined_written = True

        if prefix in self.prefix_files:
            value_line = text_value.strip()
            if self.prefix_written[prefix]:
                self.prefix_files[prefix].write("\n")
            self.prefix_files[prefix].write(value_line)
            self.prefix_written[prefix] = True


def download_file(url: str, filename: str) -> None:
    """Download a file from a given URL with a progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True)
    with open(filename, "wb") as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()
    if total_size != 0 and progress_bar.n != total_size:
        print("Error: Download incomplete.")
    else:
        print(f"Downloaded {filename}")


def find_csv_links(url: str) -> List[str]:
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    links = [
        "https://huggingface.co" + a["href"]
        for a in soup.find_all("a", href=True)
        if a["href"].endswith(".csv?download=true")
    ]
    return links


def sanitize_prefix(prefix: str) -> str:
    sanitized = re.sub(r"\s+", "_", prefix)
    sanitized = re.sub(r"[^\w.-]", "_", sanitized)
    return sanitized.strip("_")


def prepare_prefix_files(
    value_prefixes: Iterable[str], output_text_file: str
) -> Dict[str, TextIO]:
    base_dir = os.path.dirname(os.path.abspath(output_text_file)) or "."
    os.makedirs(base_dir, exist_ok=True)
    prefix_file_handles = {}
    used_names = set()
    for index, prefix in enumerate(value_prefixes, start=1):
        if prefix in prefix_file_handles:
            continue
        sanitized = sanitize_prefix(prefix)
        if not sanitized:
            sanitized = f"value_prefix_{index}"
        candidate = sanitized
        suffix = 1
        while candidate in used_names:
            candidate = f"{sanitized}_{suffix}"
            suffix += 1
        used_names.add(candidate)
        file_path = os.path.join(base_dir, f"{candidate}.txt")
        prefix_file_handles[prefix] = open(file_path, "w", encoding="utf-8")
        print(f"Writing values for prefix '{prefix}' to {file_path}")
    return prefix_file_handles


def normalize_value_prefixes(
    include_keys: List[str], value_prefixes: Optional[List[str]]
) -> List[str]:
    if not value_prefixes:
        return [""] * len(include_keys)
    if len(value_prefixes) == 1 and len(include_keys) > 1:
        return value_prefixes * len(include_keys)
    if len(value_prefixes) != len(include_keys):
        raise ValueError(
            "Number of value prefixes must match number of include keys unless a single prefix is provided."
        )
    return value_prefixes


def build_excluded_pairs(exclude: Optional[List[List[str]]]) -> Dict[str, str]:
    excluded_pairs: Dict[str, str] = {}
    if not exclude:
        return excluded_pairs
    for pair in exclude:
        if len(pair) % 2 != 0:
            raise ValueError(
                "Exclude arguments must be provided in KEY VALUE pairs."
            )
        for i in range(0, len(pair), 2):
            key = pair[i]
            value = pair[i + 1]
            excluded_pairs[key] = value
    return excluded_pairs


def emit_csv_contents(
    csv_path: str,
    include_keys: List[str],
    value_prefixes: List[str],
    required_key: Optional[str],
    skip_empty: bool,
    excluded_pairs: Dict[str, str],
    output_manager: OutputManager,
) -> None:
    print(f"Processing {csv_path}")
    with open(csv_path, "r", encoding="utf-8", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if required_key and (row.get(required_key, "") == ""):
                continue
            skip_row = False
            for ex_key, ex_value in excluded_pairs.items():
                if row.get(ex_key) == ex_value:
                    skip_row = True
                    break
            if skip_row:
                continue
            for key, prefix in zip(include_keys, value_prefixes):
                if key not in row:
                    continue
                value = row.get(key, "")
                if skip_empty and value == "":
                    continue
                output_manager.write(prefix, value)


def main(
    url: Optional[str],
    output_text_file: str,
    no_output_text: bool,
    include_keys: List[str],
    value_prefixes: Optional[List[str]],
    required_key: Optional[str],
    skip_empty: bool,
    exclude: Optional[List[List[str]]],
    direct_csv_input: Optional[List[str]],
    split_by_prefix: bool,
) -> None:
    normalized_prefixes = normalize_value_prefixes(include_keys, value_prefixes)
    excluded_pairs = build_excluded_pairs(exclude)

    if direct_csv_input:
        csv_paths = direct_csv_input
    else:
        if not url:
            raise ValueError("Either a URL or --direct_csv_input must be provided.")
        csv_links = find_csv_links(url)
        if not csv_links:
            raise ValueError(f"No CSV files found at {url}")
        download_dir = "./downloaded_csvs"
        os.makedirs(download_dir, exist_ok=True)
        csv_paths = []
        for link in csv_links:
            file_name = link.split("/")[-1].split("?")[0]
            file_path = os.path.join(download_dir, file_name)
            if not os.path.exists(file_path):
                download_file(link, file_path)
            csv_paths.append(file_path)

    combined_handle = None
    prefix_handles = None
    try:
        if not no_output_text:
            combined_dir = os.path.dirname(os.path.abspath(output_text_file))
            if combined_dir and not os.path.exists(combined_dir):
                os.makedirs(combined_dir, exist_ok=True)
            combined_handle = open(output_text_file, "w", encoding="utf-8")
        if split_by_prefix:
            prefix_handles = prepare_prefix_files(normalized_prefixes, output_text_file)
        output_manager = OutputManager(
            combined_handle if not no_output_text else None,
            prefix_handles,
        )
        for csv_path in csv_paths:
            emit_csv_contents(
                csv_path,
                include_keys,
                normalized_prefixes,
                required_key,
                skip_empty,
                excluded_pairs,
                output_manager,
            )
    finally:
        if combined_handle is not None:
            combined_handle.close()
        if prefix_handles:
            for handle in prefix_handles.values():
                handle.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download CSV files from a URL or process a local CSV and emit selected columns to text files.",
    )
    parser.add_argument(
        "--url",
        type=str,
        help="URL to scrape for CSV files.",
    )
    parser.add_argument(
        "-o",
        "--output_text_file",
        type=str,
        default="input.txt",
        help="Path to the combined output text file.",
    )
    parser.add_argument(
        "--no_output_text",
        action="store_true",
        help="Skip creation of the combined output text file.",
    )
    parser.add_argument(
        "-i",
        "--include_keys",
        type=str,
        nargs="+",
        required=True,
        help="List of column names to include from the CSV contents.",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="+",
        action="append",
        metavar=("KEY", "VALUE"),
        help="Specify key-value pairs to be excluded. Use the format: --exclude KEY VALUE [KEY VALUE ...]",
    )
    parser.add_argument(
        "-p",
        "--value_prefixes",
        type=str,
        nargs="+",
        help="List of prefixes to prepend to each value when writing to the combined text file.",
    )
    parser.add_argument(
        "-r",
        "--required_key",
        type=str,
        default=None,
        help="Only emit rows that have a non-empty value for this column (optional).",
    )
    parser.add_argument(
        "-s",
        "--skip_empty",
        action="store_true",
        help="Skip any value which is an empty string.",
    )
    parser.add_argument(
        "-c",
        "--direct_csv_input",
        type=str,
        nargs="+",
        help="Skip download and process the provided CSV file paths directly.",
    )
    parser.add_argument(
        "--split_by_prefix",
        action="store_true",
        help="Write each prefix's values to its own text file named after the prefix.",
    )

    args = parser.parse_args()

    main(
        args.url,
        args.output_text_file,
        args.no_output_text,
        args.include_keys,
        args.value_prefixes,
        args.required_key,
        args.skip_empty,
        args.exclude,
        args.direct_csv_input,
        args.split_by_prefix,
    )
