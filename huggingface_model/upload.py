from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel

config = GPT2Config.from_pretrained("gpt2-custom")
config.push_to_hub("custom_gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2-custom")
model.push_to_hub("custom_gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-custom")
tokenizer.push_to_hub("custom_gpt2")