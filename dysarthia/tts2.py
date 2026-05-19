import torch
import os
import soundfile as sf
import numpy as np

device = torch.device("cpu")
print(f"Using device: {device}")

# Load Silero Hindi TTS (no reference audio needed)
try:
    print("Loading Silero Hindi model...")
    # Load Silero Indic model (supports Hindi)
    model, _, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-models",
        model="silero_tts",
        language="indic",           # 'indic' covers Hindi
        speaker="v3_indic"
    )
    model.to(device)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Generate speech directly from text
try:
    text = "नमस्ते! संगीत की तरह जीवन भी खूबसूरत होता है, बस इसे सही ताल में जीना आना चाहिए."
    print("Generating audio...")
    audio = model.apply_tts(
        text=text,
        speaker="indic_2",          # Voice variant (hi_0, hi_1, ...)
        sample_rate=24000
    )
    print("Audio generated!")
    
    # Save
    os.makedirs("samples", exist_ok=True)
    sf.write("samples/namaste.wav", np.array(audio, dtype=np.float32), samplerate=24000)
    print("Audio saved to samples/namaste.wav")
    
except Exception as e:
    print(f"Error generating audio: {e}")