from datasets import load_dataset, Audio
import random
import re
import json


def choose_word(words):
    valid_indices = [i for i, w in enumerate(words) if w.isalpha() and len(w) > 2]
    return random.choice(valid_indices) if valid_indices else None


def sound_repetition(text):
    words = text.split()
    idx = choose_word(words)
    if idx is None:
        return text
    
    word = words[idx]
    first_letter = word[0]
    repeat_count = random.randint(2, 3)
    
    pattern = "-".join([first_letter] * repeat_count)
    words[idx] = f"{pattern}-{word}"
    
    return " ".join(words)


def syllable_repetition(text):
    words = text.split()
    idx = choose_word(words)
    if idx is None:
        return text
    
    word = words[idx]
    syllable = word[:2]  # simple heuristic
    repeat_count = random.randint(2, 3)
    
    pattern = "-".join([syllable] * repeat_count)
    words[idx] = f"{pattern}-{word}"
    
    return " ".join(words)

def prolongation(text):
    words = text.split()
    idx = choose_word(words)
    if idx is None:
        return text
    
    word = words[idx]
    char = word[0]
    stretch = char * random.randint(4, 8)
    
    words[idx] = f"{stretch}{word[1:]}"
    return " ".join(words)

def block(text):
    words = text.split()
    idx = choose_word(words)
    if idx is None:
        return text
    
    words.insert(idx, "—")
    return " ".join(words)

def interjection(text):
    fillers = ["um", "uh", "like", "you know"]
    words = text.split()
    idx = random.randint(1, len(words)-1)
    
    words.insert(idx, random.choice(fillers))
    return " ".join(words)

def phrase_repetition(text):
    words = text.split()
    if len(words) < 4:
        return text
    
    start = random.randint(0, len(words)//2)
    end = min(len(words), start + random.randint(2, 4))
    
    phrase = words[start:end]
    words = words[:end] + phrase + words[end:]
    
    return " ".join(words)

def cluttering(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = text.replace(" ", "")
    return text

def normal(text):
    return text

def apply_dysfluency(text, label):
    functions = {
        "sound_repetition": sound_repetition,
        "syllable_repetition": syllable_repetition,
        "prolongation": prolongation,
        "block": block,
        "interjection": interjection,
        "phrase_repetition": phrase_repetition,
        "cluttering": cluttering,
        "normal": normal
    }
    
    return functions[label](text)

functions = {
        "sound_repetition": sound_repetition,
        "syllable_repetition": syllable_repetition,
        "prolongation": prolongation,
        "block": block,
        "interjection": interjection,
        "phrase_repetition": phrase_repetition,
        "cluttering": cluttering,
        "normal": normal
    }

# for func in functions:
#     print(functions[func]("I want to go to school."))

#Load Dataset Belowwwww



def generate_dataset(samples_per_class=20000,output_file="dataset.jsonl"):
    ds = load_dataset(
    "openslr/librispeech_asr",
    "other",
    split="train.500",
    streaming=True
    )
    
    ds = ds.cast_column("audio", Audio(decode=False))
    
    functions = {
        "sound_repetition": sound_repetition,
        "syllable_repetition": syllable_repetition,
        "prolongation": prolongation,
        "block": block,
        "interjection": interjection,
        "phrase_repetition": phrase_repetition,
        "cluttering": cluttering,
        "normal": normal
    }
    
    labels=list(functions.keys())
    counts={label:0 for label in labels}
    total_needed=samples_per_class*len(labels)
    
    with open(output_file,"w",encoding="utf-8") as f:
        for example in ds:
            base_text = example["text"]

            # Skip short sentences
            if len(base_text.split()) < 5:
                continue

            # Choose label that still needs samples
            available_labels = [l for l in labels if counts[l] < samples_per_class]
            
            if not available_labels:
                break

            label = random.choice(available_labels)
            new_text = functions[label](base_text)

            f.write(json.dumps({
                "text": new_text,
                "label": label
            },ensure_ascii=False) + "\n")

            counts[label] += 1

        print("Generation complete.")
        print("Counts:", counts)
    
generate_dataset()