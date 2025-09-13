from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained("./merged-mistral7b", torch_dtype="auto", device_map="auto")
t = AutoTokenizer.from_pretrained("./merged-mistral7b")
print("Loaded OK. Vocab:", t.vocab_size)