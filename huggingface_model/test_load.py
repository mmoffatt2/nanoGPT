from transformers import AutoConfig, AutoModel, AutoTokenizer, pipeline
config = AutoConfig.from_pretrained("mmoffatt/custom_gpt2", revision="fabe18eca9deb456bbea6ca5edaecc59d7e36cf7")
model = AutoModel.from_pretrained("mmoffatt/custom_gpt2", revision="fabe18eca9deb456bbea6ca5edaecc59d7e36cf7")
tokenizer = AutoTokenizer.from_pretrained("mmoffatt/custom_gpt2", revision="fabe18eca9deb456bbea6ca5edaecc59d7e36cf7")

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
output = generator("Once upon a time", max_length=50, num_return_sequences=1)
print(output)