import io
import torch
import torchaudio
import pandas as pd
from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_id = "openai/whisper-large-v2"

processor = WhisperProcessor.from_pretrained(model_id, language="sw", task="transcribe")

model = WhisperForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16
).to(device)
model.eval()

ds = load_dataset("sartifyllc/Sartify_ITU_Zindi_Testdataset", split="test")
ds = ds.cast_column("audio", Audio(decode=False))

def decode_audio_bytes(audio_bytes):
    with io.BytesIO(audio_bytes) as f:
        waveform, sr = torchaudio.load(f)
    return waveform.squeeze(0), sr

def transcribe(audio_bytes):
    waveform, sr = decode_audio_bytes(audio_bytes)

    if waveform.abs().mean() < 1e-4 or waveform.shape[0] < 16000:
        return "[SKIPPED]"

    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    inputs = processor.feature_extractor(
        waveform.numpy(), sampling_rate=16000, return_tensors="pt"
    ).input_features.to(device)

    forced_ids = processor.get_decoder_prompt_ids(language="sw", task="transcribe")

    with torch.no_grad():
        predicted_ids = model.generate(
            inputs,
            forced_decoder_ids=forced_ids,
            num_beams=5,
            length_penalty=1.0,
            max_new_tokens=128,
            repetition_penalty=1.2,
            no_repeat_ngram_size=2
        )

    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].lower().strip()

results = []
print("Transcribing dataset...")

for i, sample in enumerate(ds):
    filename = sample["filename"]
    audio_bytes = sample["audio"]["bytes"]

    try:
        text = transcribe(audio_bytes)
        print(f"{i+1}/{len(ds)} | {filename} → {text}")
    except Exception as e:
        print(f"Error on {filename}: {e}")
        text = "[ERROR]"

    results.append({"filename": filename, "text": text})

    if (i + 1) % 100 == 0:
        pd.DataFrame(results).to_csv(f"submission_partial_{i+1}.csv", index=False)
        print(f"Auto-saved at {i+1} samples")

pd.DataFrame(results).to_csv("", index=False) # add the path
print("saved as 'submission_whisper_large.csv'")
