from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("AdaptLLM/medicine-chat")
tokenizer = AutoTokenizer.from_pretrained("AdaptLLM/medicine-chat")

# Put your input here:
user_input = '''Question: Which of the following is an example of monosomy?
Options:
- 46,XX
- 47,XXX
- 69,XYY
- 45,X

Please provide your choice first and then provide explanations if possible.'''

# Apply the prompt template and system prompt of LLaMA-2-Chat demo for chat models (NOTE: NO prompt template is required for base models!)
our_system_prompt = "\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n" # Please do NOT change this
prompt = f"<s>[INST] <<SYS>>{our_system_prompt}<</SYS>>\n\n{user_input} [/INST]"

# # NOTE:
# # If you want to apply your own system prompt, please integrate it into the instruction part following our system prompt like this:
# your_system_prompt = "Please, answer this question faithfully."
# prompt = f"<s>[INST] <<SYS>>{our_system_prompt}<</SYS>>\n\n{your_system_prompt}\n{user_input} [/INST]"

inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
outputs = model.generate(input_ids=inputs, max_length=4096)[0]

answer_start = int(inputs.shape[-1])
pred = tokenizer.decode(outputs[answer_start:], skip_special_tokens=True)

print(f'### User Input:\n{user_input}\n\n### Assistant Output:\n{pred}')
