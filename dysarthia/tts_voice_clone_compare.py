"""
Text-to-Speech with Coqui TTS (Voice Cloning) + Audio Comparison Suite
=======================================================================
Dependencies (install once):
    pip install TTS librosa soundfile numpy scipy matplotlib praat-parselmouth

Usage:
    python tts_voice_clone_compare.py --ref_audio path/to/reference.wav

The script will:
  1. Generate speech using XTTS-v2 (voice-cloning from reference audio)
  2. Compare generated audio vs reference across:
       - Waveform cross-correlation
       - MFCCs cosine similarity
       - Mel Cepstral Distortion (MCD)
       - Pitch / F0 comparison
  3. Print a full report and save a comparison plot
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import librosa
import soundfile as sf
from scipy.signal import correlate
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─── CONFIG ──────────────────────────────────────────────────────────────────

TEXT_TO_SPEAK = (
    "The quick brown fox jumps over the lazy dog. "
    "Voice cloning technology has come a remarkably long way in recent years."
)

OUTPUT_AUDIO  = "generated_speech.wav"
PLOT_OUTPUT   = "audio_comparison.png"

SAMPLE_RATE   = 22050   # XTTS-v2 native SR
N_MFCC        = 40
HOP_LENGTH    = 256
N_FFT         = 1024

# ─── STEP 1 – GENERATE TTS ───────────────────────────────────────────────────

def generate_tts(reference_audio: str, output_path: str) -> str:
    """Use Coqui XTTS-v2 to clone the reference voice and synthesise TEXT_TO_SPEAK."""
    print("\n[1/3] Loading Coqui XTTS-v2 model …")
    from TTS.api import TTS                          # lazy import after deps check

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

    print(f"[1/3] Synthesising with voice cloned from: {reference_audio}")
    tts.tts_to_file(
        text=TEXT_TO_SPEAK,
        speaker_wav=reference_audio,
        language="en",
        file_path=output_path,
    )
    print(f"[1/3] Generated audio saved → {output_path}")
    return output_path


# ─── STEP 2 – LOAD AUDIO ─────────────────────────────────────────────────────

def load_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


# ─── STEP 3 – METRICS ────────────────────────────────────────────────────────

# 3a. Waveform cross-correlation
def waveform_correlation(ref: np.ndarray, gen: np.ndarray) -> dict:
    """Normalised cross-correlation; peak value ∈ [-1, 1]."""
    # Pad shorter signal
    n = max(len(ref), len(gen))
    r = np.pad(ref, (0, n - len(ref)))
    g = np.pad(gen, (0, n - len(gen)))

    corr   = correlate(r, g, mode="full")
    norm   = np.sqrt(np.sum(r**2) * np.sum(g**2)) + 1e-9
    corr_n = corr / norm

    peak_val = float(np.max(np.abs(corr_n)))
    peak_lag = int(np.argmax(np.abs(corr_n))) - (n - 1)   # samples

    return {
        "peak_correlation": round(peak_val, 6),
        "peak_lag_samples": peak_lag,
        "peak_lag_ms":      round(peak_lag / SAMPLE_RATE * 1000, 2),
    }


# 3b. MFCC cosine similarity
def mfcc_similarity(ref: np.ndarray, gen: np.ndarray) -> dict:
    """Compare mean MFCC vectors with cosine similarity (1 = identical)."""
    mfcc_ref = librosa.feature.mfcc(y=ref, sr=SAMPLE_RATE,
                                    n_mfcc=N_MFCC, hop_length=HOP_LENGTH, n_fft=N_FFT)
    mfcc_gen = librosa.feature.mfcc(y=gen, sr=SAMPLE_RATE,
                                    n_mfcc=N_MFCC, hop_length=HOP_LENGTH, n_fft=N_FFT)

    mean_ref = np.mean(mfcc_ref, axis=1)
    mean_gen = np.mean(mfcc_gen, axis=1)

    sim = 1.0 - cosine(mean_ref, mean_gen)

    # Per-coefficient delta (shows which cepstral bands differ most)
    delta = np.abs(mean_ref - mean_gen)

    return {
        "cosine_similarity":      round(float(sim), 6),
        "mean_mfcc_ref":          mean_ref.tolist(),
        "mean_mfcc_gen":          mean_gen.tolist(),
        "per_coeff_abs_delta":    delta.tolist(),
        "max_delta_coeff_index":  int(np.argmax(delta)),
        "max_delta_value":        round(float(np.max(delta)), 4),
    }


# 3c. Mel Cepstral Distortion (MCD) – industry standard for TTS evaluation
def mel_cepstral_distortion(ref: np.ndarray, gen: np.ndarray) -> dict:
    """
    MCD = (10 / ln(10)) * mean_frame( sqrt(2 * sum((c_ref - c_gen)^2)) )
    Lower is better; < 8 dB is considered good for voice conversion.
    """
    mfcc_ref = librosa.feature.mfcc(y=ref, sr=SAMPLE_RATE,
                                    n_mfcc=N_MFCC, hop_length=HOP_LENGTH, n_fft=N_FFT)
    mfcc_gen = librosa.feature.mfcc(y=gen, sr=SAMPLE_RATE,
                                    n_mfcc=N_MFCC, hop_length=HOP_LENGTH, n_fft=N_FFT)

    # Align lengths (truncate to shorter)
    min_frames = min(mfcc_ref.shape[1], mfcc_gen.shape[1])
    mfcc_ref   = mfcc_ref[:, :min_frames]
    mfcc_gen   = mfcc_gen[:, :min_frames]

    diff      = mfcc_ref - mfcc_gen
    frame_mcd = np.sqrt(2 * np.sum(diff**2, axis=0))   # per frame
    mcd       = (10.0 / np.log(10)) * np.mean(frame_mcd)

    return {
        "mcd_db":            round(float(mcd), 4),
        "mcd_interpretation": (
            "Excellent (< 4 dB)" if mcd < 4 else
            "Good (4–8 dB)"      if mcd < 8 else
            "Fair (8–12 dB)"     if mcd < 12 else
            "Poor (> 12 dB)"
        ),
        "frames_compared":   min_frames,
    }


# 3d. Pitch / F0 comparison
def pitch_comparison(ref: np.ndarray, gen: np.ndarray) -> dict:
    """Extract fundamental frequency (F0) via pyin and compare statistics."""
    f0_ref, voiced_ref, _ = librosa.pyin(
        ref, fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=SAMPLE_RATE, hop_length=HOP_LENGTH)

    f0_gen, voiced_gen, _ = librosa.pyin(
        gen, fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=SAMPLE_RATE, hop_length=HOP_LENGTH)

    # Only voiced frames
    f0_ref_v = f0_ref[voiced_ref & ~np.isnan(f0_ref)]
    f0_gen_v = f0_gen[voiced_gen & ~np.isnan(f0_gen)]

    def safe_stats(arr):
        if len(arr) == 0:
            return {"mean": None, "std": None, "median": None, "min": None, "max": None}
        return {
            "mean":   round(float(np.mean(arr)),   2),
            "std":    round(float(np.std(arr)),    2),
            "median": round(float(np.median(arr)), 2),
            "min":    round(float(np.min(arr)),    2),
            "max":    round(float(np.max(arr)),    2),
        }

    stats_ref = safe_stats(f0_ref_v)
    stats_gen = safe_stats(f0_gen_v)

    mean_diff = (
        round(abs(stats_ref["mean"] - stats_gen["mean"]), 2)
        if stats_ref["mean"] and stats_gen["mean"] else None
    )

    return {
        "reference_f0_hz":   stats_ref,
        "generated_f0_hz":   stats_gen,
        "mean_f0_diff_hz":   mean_diff,
        "voiced_ratio_ref":  round(float(np.sum(voiced_ref) / max(len(voiced_ref), 1)), 4),
        "voiced_ratio_gen":  round(float(np.sum(voiced_gen) / max(len(voiced_gen), 1)), 4),
        "f0_ref_voiced_frames": len(f0_ref_v),
        "f0_gen_voiced_frames": len(f0_gen_v),
        "_f0_ref_raw": f0_ref,   # kept for plotting (not printed)
        "_f0_gen_raw": f0_gen,
    }


# ─── STEP 4 – REPORT ─────────────────────────────────────────────────────────

def print_report(corr: dict, mfcc: dict, mcd: dict, pitch: dict) -> None:
    div = "─" * 60
    print(f"\n{div}")
    print("  AUDIO COMPARISON REPORT")
    print(div)

    print("\n📊  WAVEFORM CROSS-CORRELATION")
    print(f"    Peak Normalised Correlation : {corr['peak_correlation']:.4f}  (1.0 = perfect)")
    print(f"    Peak Lag                    : {corr['peak_lag_ms']} ms  ({corr['peak_lag_samples']} samples)")

    print("\n🎙️  MFCC COSINE SIMILARITY")
    print(f"    Cosine Similarity           : {mfcc['cosine_similarity']:.4f}  (1.0 = identical)")
    print(f"    Highest-delta MFCC coeff    : #{mfcc['max_delta_coeff_index']}  (Δ = {mfcc['max_delta_value']})")

    print("\n🎵  MEL CEPSTRAL DISTORTION (MCD)")
    print(f"    MCD                         : {mcd['mcd_db']} dB")
    print(f"    Quality Grade               : {mcd['mcd_interpretation']}")
    print(f"    Frames compared             : {mcd['frames_compared']}")

    print("\n🎶  PITCH / F0 COMPARISON")
    r, g = pitch["reference_f0_hz"], pitch["generated_f0_hz"]
    print(f"    Reference  – Mean: {r['mean']} Hz  |  Std: {r['std']} Hz  |  Median: {r['median']} Hz")
    print(f"    Generated  – Mean: {g['mean']} Hz  |  Std: {g['std']} Hz  |  Median: {g['median']} Hz")
    print(f"    Mean F0 Difference          : {pitch['mean_f0_diff_hz']} Hz")
    print(f"    Voiced Ratio (ref/gen)      : {pitch['voiced_ratio_ref']:.2%} / {pitch['voiced_ratio_gen']:.2%}")

    # Aggregate score (simple weighted heuristic for quick overview)
    score = (
        corr["peak_correlation"] * 0.25 +
        mfcc["cosine_similarity"] * 0.35 +
        max(0, 1 - mcd["mcd_db"] / 20) * 0.25 +
        (1 - min(1, (pitch["mean_f0_diff_hz"] or 20) / 100)) * 0.15
    ) * 100

    print(f"\n⭐  AGGREGATE SIMILARITY SCORE  :  {score:.1f} / 100")
    print(div)


# ─── STEP 5 – PLOT ───────────────────────────────────────────────────────────

def save_plot(ref: np.ndarray, gen: np.ndarray,
              mfcc_data: dict, pitch_data: dict, output: str) -> None:
    """Save a 4-panel comparison figure."""
    fig = plt.figure(figsize=(16, 12), facecolor="#0f0f0f")
    gs  = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)
    plt.suptitle("TTS vs Reference – Audio Comparison", color="white",
                 fontsize=15, fontweight="bold", y=0.98)

    t_ref = np.linspace(0, len(ref) / SAMPLE_RATE, len(ref))
    t_gen = np.linspace(0, len(gen) / SAMPLE_RATE, len(gen))

    # --- 1. Waveforms ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#1a1a2e")
    ax1.plot(t_ref, ref, color="#4fc3f7", linewidth=0.5, alpha=0.85, label="Reference")
    ax1.plot(t_gen, gen, color="#ef9a9a", linewidth=0.5, alpha=0.85, label="Generated")
    ax1.set_title("Waveform Overlay", color="white")
    ax1.set_xlabel("Time (s)", color="grey"); ax1.set_ylabel("Amplitude", color="grey")
    ax1.tick_params(colors="grey"); ax1.legend(facecolor="#222", labelcolor="white")
    for spine in ax1.spines.values(): spine.set_edgecolor("#444")

    # --- 2. MFCC delta heatmap ---
    mfcc_ref_mat = librosa.feature.mfcc(y=ref, sr=SAMPLE_RATE,
                                         n_mfcc=N_MFCC, hop_length=HOP_LENGTH, n_fft=N_FFT)
    mfcc_gen_mat = librosa.feature.mfcc(y=gen, sr=SAMPLE_RATE,
                                         n_mfcc=N_MFCC, hop_length=HOP_LENGTH, n_fft=N_FFT)
    min_f = min(mfcc_ref_mat.shape[1], mfcc_gen_mat.shape[1])
    delta_map = np.abs(mfcc_ref_mat[:, :min_f] - mfcc_gen_mat[:, :min_f])

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#1a1a2e")
    im = ax2.imshow(delta_map, aspect="auto", origin="lower", cmap="magma",
                    extent=[0, min_f * HOP_LENGTH / SAMPLE_RATE, 0, N_MFCC])
    plt.colorbar(im, ax=ax2, label="Absolute Δ MFCC")
    ax2.set_title("MFCC Absolute Difference Map", color="white")
    ax2.set_xlabel("Time (s)", color="grey"); ax2.set_ylabel("MFCC Coeff", color="grey")
    ax2.tick_params(colors="grey")
    for spine in ax2.spines.values(): spine.set_edgecolor("#444")

    # --- 3. Per-coefficient MFCC bar ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("#1a1a2e")
    coeffs   = np.arange(N_MFCC)
    mean_ref = np.array(mfcc_data["mean_mfcc_ref"])
    mean_gen = np.array(mfcc_data["mean_mfcc_gen"])
    w = 0.38
    ax3.bar(coeffs - w/2, mean_ref, width=w, color="#4fc3f7", label="Reference", alpha=0.85)
    ax3.bar(coeffs + w/2, mean_gen, width=w, color="#ef9a9a", label="Generated", alpha=0.85)
    ax3.set_title("Mean MFCC per Coefficient", color="white")
    ax3.set_xlabel("Coefficient Index", color="grey"); ax3.set_ylabel("Mean Value", color="grey")
    ax3.tick_params(colors="grey"); ax3.legend(facecolor="#222", labelcolor="white")
    for spine in ax3.spines.values(): spine.set_edgecolor("#444")

    # --- 4. F0 pitch track ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("#1a1a2e")
    f0_ref_raw = pitch_data["_f0_ref_raw"]
    f0_gen_raw = pitch_data["_f0_gen_raw"]
    t_p_ref = np.arange(len(f0_ref_raw)) * HOP_LENGTH / SAMPLE_RATE
    t_p_gen = np.arange(len(f0_gen_raw)) * HOP_LENGTH / SAMPLE_RATE
    ax4.plot(t_p_ref, f0_ref_raw, color="#4fc3f7", linewidth=1.2, label="Reference F0", alpha=0.85)
    ax4.plot(t_p_gen, f0_gen_raw, color="#ef9a9a", linewidth=1.2, label="Generated F0", alpha=0.85)
    ax4.set_title("Pitch Contour (F0)", color="white")
    ax4.set_xlabel("Time (s)", color="grey"); ax4.set_ylabel("Frequency (Hz)", color="grey")
    ax4.tick_params(colors="grey"); ax4.legend(facecolor="#222", labelcolor="white")
    for spine in ax4.spines.values(): spine.set_edgecolor("#444")

    plt.savefig(output, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n[3/3] Comparison plot saved → {output}")
    plt.close()


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Coqui TTS Voice Clone + Audio Comparison")
    parser.add_argument("--ref_audio", required=True,
                        help="Path to reference .wav file (used for voice cloning)")
    parser.add_argument("--output",    default=OUTPUT_AUDIO,
                        help=f"Output .wav path (default: {OUTPUT_AUDIO})")
    parser.add_argument("--plot",      default=PLOT_OUTPUT,
                        help=f"Comparison plot path (default: {PLOT_OUTPUT})")
    parser.add_argument("--skip_tts",  action="store_true",
                        help="Skip TTS generation (use existing --output file for comparison only)")
    args = parser.parse_args()

    if not os.path.exists(args.ref_audio):
        raise FileNotFoundError(f"Reference audio not found: {args.ref_audio}")

    # 1. Generate
    if not args.skip_tts:
        generate_tts(args.ref_audio, args.output)
    else:
        print(f"[1/3] Skipping TTS – using existing file: {args.output}")

    # 2. Load
    print("\n[2/3] Loading audio files for comparison …")
    ref = load_audio(args.ref_audio)
    gen = load_audio(args.output)
    print(f"      Reference : {len(ref)/SAMPLE_RATE:.2f}s  |  Generated : {len(gen)/SAMPLE_RATE:.2f}s")

    # 3. Compute metrics
    print("      Computing metrics …")
    corr  = waveform_correlation(ref, gen)
    mfcc  = mfcc_similarity(ref, gen)
    mcd   = mel_cepstral_distortion(ref, gen)
    pitch = pitch_comparison(ref, gen)

    # 4. Report
    print_report(corr, mfcc, mcd, pitch)

    # 5. Plot
    save_plot(ref, gen, mfcc, pitch, args.plot)

    print("\n✅  Done.\n")


if __name__ == "__main__":
    main()
