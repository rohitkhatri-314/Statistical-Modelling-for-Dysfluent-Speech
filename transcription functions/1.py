from datasets import load_dataset
ds = load_dataset("openslr/librispeech_asr", "clean", split="train.100")
texts = ds["text"]

print(type(texts))