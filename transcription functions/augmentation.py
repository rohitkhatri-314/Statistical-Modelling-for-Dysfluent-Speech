import random
import re


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

    
print(sound_repetition("I want to go to school"))