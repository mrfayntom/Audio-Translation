# ASR Transcription and Grammar Correction for Swahili

This project involves transcribing **Swahili audio files** to text using an **Automatic Speech Recognition (ASR)** model and then applying **grammar correction** to improve the quality of the transcriptions.

## Overview

The script utilizes the **Mollel/ASR-Swahili-Small** model from Hugging Face for speech-to-text transcription. After the transcription is done, a grammar correction step is applied to fix common errors and improve the quality of the output.

### Key Steps:
1. **Audio Transcription**: Convert Swahili audio to text using the Mollel ASR model.
2. **Grammar Correction**: Apply grammar fixes to the transcribed text for improved accuracy.
3. **Output**: Save the transcriptions (and corrected text) to a CSV file for further use or analysis.

## Model Used

- **Mollel/ASR-Swahili-Small**: A pre-trained automatic speech recognition (ASR) model fine-tuned for Swahili. This model converts Swahili speech in audio files to text.

## Requirements

To run this project, the following dependencies must be installed:

- Python 3.x
- PyTorch
- torchaudio
- pandas
- Hugging Face `transformers` and `datasets`
- `difflib` (for word similarity check)
- CUDA (for GPU support)

You can install the required packages using pip:

```bash
pip install torch torchaudio pandas datasets transformers difflib
```
## Personal Note

I sincerely apologize for not being able to dedicate more time to my coding journey lately. At the moment, I’m fully focused on preparing for the JEE Mains, which demands a lot of my attention and energy. As a result, my coding progress has been slower than usual. On top of that, I’m also working on finding a solid startup idea that could potentially help fund my college fees. Balancing these priorities has been challenging, but I’m hopeful that with some perseverance, I’ll be able to get back on track with my coding goals soon.

Thank you for understanding, and I appreciate your patience!
