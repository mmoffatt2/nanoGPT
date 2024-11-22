# tests.py

import unittest
import os
from tokenizers import (
    NumericRangeTokenizer,
    SentencePieceTokenizer,
    TiktokenTokenizer,
    CustomTokenizer,
    CharTokenizer,
)
from argparse import Namespace
from rich.console import Console
from rich.theme import Theme
from rich.table import Table

console = Console(theme=Theme({
    "pass": "bold green",
    "fail": "bold red",
    "test_name": "bold yellow",
    "separator": "grey50",
    "input": "bold cyan",
    "output": "bold magenta",
    "info": "bold blue"
}))


class RichTestResult(unittest.TestResult):
    def __init__(self):
        super().__init__()
        self.test_results = []

    def addSuccess(self, test):
        self.test_results.append((test, 'PASS'))
        console.print("[bold green]Test Passed.[/bold green]")
        super().addSuccess(test)

    def addFailure(self, test, err):
        self.test_results.append((test, 'FAIL'))
        console.print("[bold red]Test Failed.[/bold red]")
        super().addFailure(test, err)

    def addError(self, test, err):
        self.test_results.append((test, 'FAIL'))
        console.print("[bold red]Test Error.[/bold red]")
        super().addError(test, err)

    def startTest(self, test):
        console.print('-' * 80, style='separator')
        console.print(f"Running test: [bold]{test._testMethodName}[/bold]", style='test_name')
        super().startTest(test)

    def stopTest(self, test):
        super().stopTest(test)


def run_tests():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestTokenizers)
    result = RichTestResult()
    suite.run(result)
    # Print final table
    console.print('=' * 80, style='separator')
    console.print("[bold]Test Results:[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Test")
    table.add_column("Result", justify="center")
    for test, status in result.test_results:
        test_name = test._testMethodName
        if status == 'PASS':
            style = "pass"
        else:
            style = "fail"
        table.add_row(test_name, f"[{style}]{status}[/{style}]")
    console.print(table)


class TestTokenizers(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.sample_text = "Hello, world!\nThis is a test."
        self.numeric_data = "123\n456\n789"
        self.tokens_file = "tokens.txt"

        # Create a tokens file for custom tokenizers
        with open(self.tokens_file, 'w') as f:
            f.write("Hello\nworld\nThis\nis\na\ntest\n")

    def tearDown(self):
        # Clean up tokens file
        if os.path.exists(self.tokens_file):
            os.remove(self.tokens_file)
        # Remove temporary files created by SentencePiece
        for fname in ["spm_input.txt", "trained_spm_model"]:
            for ext in ["", ".model", ".vocab"]:
                full_name = f"{fname}{ext}"
                if os.path.exists(full_name):
                    os.remove(full_name)
        if os.path.exists("meta.pkl"):
            os.remove("meta.pkl")
        if os.path.exists("remaining.txt"):
            os.remove("remaining.txt")

    def test_numeric_range_tokenizer(self):
        args = Namespace(min_token=100, max_token=1000)
        tokenizer = NumericRangeTokenizer(args)
        ids = tokenizer.tokenize(self.numeric_data)
        detokenized = tokenizer.detokenize(ids)

        console.print("[input]Input:[/input]")
        console.print(self.numeric_data.strip(), style="input")
        console.print("[output]Detokenized Output:[/output]")
        console.print(detokenized, style="output")

        self.assertEqual(self.numeric_data.strip(), detokenized)

    def test_sentencepiece_tokenizer(self):
        args = Namespace(
            vocab_size=30,
            spm_model_file=None,
            spm_vocab_file=None,
            skip_tokenization=False
        )
        # Simulate training data
        with open("spm_input.txt", "w") as f:
            f.write(self.sample_text)
        tokenizer = SentencePieceTokenizer(args, input_files="spm_input.txt")
        ids = tokenizer.tokenize(self.sample_text)
        detokenized = tokenizer.detokenize(ids)

        console.print("[input]Input:[/input]")
        console.print(self.sample_text, style="input")
        console.print("[output]Detokenized Output:[/output]")
        console.print(detokenized, style="output")

        self.assertEqual(self.sample_text, detokenized)

    def test_tiktoken_tokenizer(self):
        args = Namespace(tiktoken_encoding='gpt2')
        tokenizer = TiktokenTokenizer(args)
        ids = tokenizer.tokenize(self.sample_text)
        detokenized = tokenizer.detokenize(ids)

        console.print("[input]Input:[/input]")
        console.print(self.sample_text, style="input")
        console.print("[output]Detokenized Output:[/output]")
        console.print(detokenized, style="output")

        self.assertEqual(self.sample_text, detokenized)

    def test_custom_tokenizer(self):
        args = Namespace(tokens_file=self.tokens_file)
        tokenizer = CustomTokenizer(args)
        ids = tokenizer.tokenize(self.sample_text)
        detokenized = tokenizer.detokenize(ids)

        console.print("[input]Input:[/input]")
        console.print(self.sample_text, style="input")
        console.print("[output]Detokenized Output:[/output]")
        console.print(detokenized, style="output")

        tokens_to_check = ["Hello", "world", "This", "is", "a", "test"]
        for token in tokens_to_check:
            self.assertIn(token, detokenized)

    def test_char_tokenizer(self):
        args = Namespace(reuse_chars=False)
        tokenizer = CharTokenizer(args, self.sample_text, None)
        ids = tokenizer.tokenize(self.sample_text)
        detokenized = tokenizer.detokenize(ids)

        console.print("[input]Input:[/input]")
        console.print(self.sample_text, style="input")
        console.print("[output]Detokenized Output:[/output]")
        console.print(detokenized, style="output")

        self.assertEqual(self.sample_text, detokenized)


if __name__ == '__main__':
    run_tests()

