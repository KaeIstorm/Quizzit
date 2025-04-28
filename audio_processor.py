import os
import ffmpeg
from faster_whisper import WhisperModel

def extract_audio_with_ffmpeg(video_path, output_audio_path="audio.wav"):
    if os.path.exists(output_audio_path):
        print(f"Audio file {output_audio_path} already exists.")
        return output_audio_path

    ffmpeg.input(video_path).output(output_audio_path).run()
    print(f"Audio saved to {output_audio_path}")
    return output_audio_path


def transcribe_audio(audio_path, model_size="tiny", cache_path="transcription.txt"):
    if os.path.exists(cache_path):
        print(f"[INFO] Loading cached transcription from {cache_path}")
        with open(cache_path, 'r', encoding='utf-8') as f:
            return f.read()

    print(f"[INFO] Loading Faster-Whisper model '{model_size}' on CUDA...")
    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    print(f"[INFO] Transcribing audio from {audio_path}...")
    segments, _ = model.transcribe(audio_path)

    transcription = " ".join([segment.text for segment in segments])

    with open(cache_path, 'w', encoding='utf-8') as f:
        f.write(transcription.strip())
    print(f"[INFO] Transcription saved to {cache_path}")

    return transcription.strip()