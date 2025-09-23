# data/template/utils/en2ipa.py

import subprocess
from konlpy.tag import Okt
import argparse
import re
import json
from typing import List
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn, MofNCompleteColumn

counter = 0

def transcribe_english(sentence, wrapper=False):
    """Transcribe an English sentence into its phonemes using espeak."""
    try:
        result = subprocess.run(
            ["espeak-ng", "-q", "-v", "en", "--ipa", sentence],
            capture_output=True,
            text=True
        )
        transcription = result.stdout.strip().replace("ㆍ", " ")
        if "(en)" in transcription:
            return f"[[[[[{sentence}]]]]]" if wrapper else sentence
        return transcription
    except Exception as e:
        return f"Error in transcribing English: {str(e)}"

def handle_mixed_language(word, wrapper=False):
    """Handle a word with potential English, Language, or number content."""
    global counter
    if word.isdigit():
        return word
    elif any('a' <= char.lower() <= 'z' for char in word):
        return transcribe_english(word, wrapper=wrapper)
    else:
        if wrapper:
            return "[[[[[" + word + "]]]]]"
        else:
            counter += 1
            return word

_WORD_RE = re.compile(r'\w+|[^\w\s]', re.UNICODE)

def transcribe_tokens_to_string(tokens: List[str], wrapper: bool) -> str:
    result = []
    for tok in tokens:
        if re.match(r'\w+', tok):
            result.append(handle_mixed_language(tok, wrapper=wrapper))
        else:
            result.append(tok)
    return " ".join(result)

def transcribe_multilingual(sentences, input_json_key=None, output_json_key='ipa', wrapper=False):
    """Transcribe multilingual sentences (JSON list mode)."""
    try:
        data = json.loads(sentences) if isinstance(sentences, str) else sentences
        if not isinstance(data, list):
            raise ValueError("JSON data should be a list of objects.")

        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=False,
        ) as progress:
            task = progress.add_task("Processing JSON items", total=len(data))
            for item in data:
                if input_json_key in item:
                    sentence = item[input_json_key]
                    tokens = _WORD_RE.findall(sentence)
                    transcription_result = transcribe_tokens_to_string(tokens, wrapper=wrapper)
                    item[output_json_key] = transcription_result
                progress.update(task, advance=1)

    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error: {e}")
        return None

    return json.dumps(data, ensure_ascii=False, indent=4)

def transcribe_text_lines(lines: List[str], wrapper: bool) -> List[str]:
    """Transcribe a plain-text file line-by-line."""
    out_lines = []
    with Progress(
        TextColumn("[bold green]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=False,
    ) as progress:
        task = progress.add_task("Processing text lines", total=len(lines))
        for line in lines:
            raw = line.rstrip("\n")
            tokens = _WORD_RE.findall(raw)
            transcribed = transcribe_tokens_to_string(tokens, wrapper=wrapper)
            out_lines.append(transcribed)
            progress.update(task, advance=1)
    return out_lines

def main():
    parser = argparse.ArgumentParser(
        description='Transcribe multilingual content into IPA phonemes. Supports JSON list mode and plain-text line mode.'
    )
    parser.add_argument('input_file', type=str, help='Path to the input file (JSON list or plain text).')
    parser.add_argument('--mode', choices=['json', 'text'], default='json',
                        help='Processing mode. "json" expects a JSON list; "text" treats file as plain text.')
    parser.add_argument('--input_json_key', type=str, help='JSON key to read sentences from (required for --mode json).')
    parser.add_argument('--output_json_key', type=str, default='ipa', help='JSON key to store IPA (default: "ipa").')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output file path for text mode. Defaults to overwriting input.')
    parser.add_argument("--wrapper", type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help="Wrap unparseable non-English text with [[[[[...]]]]] for later recovery.")
    args = parser.parse_args()

    try:
        if args.mode == 'json':
            if not args.input_json_key:
                raise ValueError("--input_json_key is required when --mode json")
            with open(args.input_file, 'r', encoding='utf-8') as f:
                input_content = f.read()
            updated_json_data = transcribe_multilingual(
                input_content,
                args.input_json_key,
                args.output_json_key,
                wrapper=args.wrapper
            )
            if updated_json_data:
                with open(args.input_file, 'w', encoding='utf-8') as f:
                    f.write(updated_json_data)
                print(f"✅ Successfully updated JSON data in '{args.input_file}'")
        else:
            with open(args.input_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            out_lines = transcribe_text_lines(lines, wrapper=args.wrapper)
            target_path = args.output_file if args.output_file else args.input_file
            with open(target_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(out_lines) + ("\n" if out_lines else ""))
            print(f"✅ Successfully wrote transcribed text to '{target_path}'")
        print(f"📊 Stats: {counter} unparseable words")
    except FileNotFoundError:
        print(f"Error: Input file '{args.input_file}' not found.")
    except ValueError as ve:
        print(f"Error: {ve}")

if __name__ == '__main__':
    main()

