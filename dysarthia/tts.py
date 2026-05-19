import torch
import os
from transformers import AutoModel
import numpy as np
import soundfile as sf

# Set device to CPU to avoid meta tensor issues
device = torch.device("cpu")
print(f"Using device: {device}")

# Check if reference audio exists
ref_audio = "test1.wav"
if not os.path.exists(ref_audio):
    print(f"Error: Reference audio {ref_audio} not found!")
    exit()

# Load IndicF5 model
repo_id = "ai4bharat/IndicF5"
try:
    print("Loading model...")
    model = AutoModel.from_pretrained(
        repo_id,
        trust_remote_code=True,
        device_map="cpu",  # Force CPU
        torch_dtype=torch.float32
    )
    model = model.to(device)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Generate speech
try:
    print("Generating audio...")
    audio = model(
        text="नमस्ते! संगीत की तरह जीवन भी खूबसूरत होता है, बस इसे सही ताल में जीना आना चाहिए.",
        ref_audio_path=ref_audio,
        ref_text="ਭਹੰਪੀ ਵਿੱਚ ਸਮਾਰਕਾਂ ਦੇ ਭਵਨ ਨਿਰਮਾਣ ਕਲਾ ਦੇ ਵੇਰਵੇ ਗੁੰਝਲਦਾਰ ਅਤੇ ਹੈਰਾਨ ਕਰਨ ਵਾਲੇ ਹਨ, ਜੋ ਮੈਨੂੰ ਖੁਸ਼ ਕरਦੇ ਹਨ।"
    )
    print("Audio generated!")
    # Normalize and save
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    os.makedirs("samples", exist_ok=True)  # Create samples folder if it doesn't exist
    sf.write("samples/namaste.wav", np.array(audio, dtype=np.float32), samplerate=24000)
    print("Audio saved to samples/namaste.wav")
except Exception as e:
    print(f"Error generating audio: {e}")