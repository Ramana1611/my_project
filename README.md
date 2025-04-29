# Real-Time Speech Processor with Web API

This project provides real-time speech processing capabilities including speech-to-text transcription, text correction, and text-to-speech synthesis. It now includes a web API backend and a frontend website for easy interaction.

## Features

- Speech-to-text transcription using Wav2Vec2
- Text correction using BERT and GPT-2
- Text-to-speech synthesis using gTTS
- Flask API backend with endpoints:
  - `/api/transcribe` - Accepts audio file and returns transcription
  - `/api/correct` - Accepts text and returns corrected text
  - `/api/speak` - Accepts text and returns speech audio
- Frontend website to record voice, display transcription and corrected text, and play synthesized speech

## Setup

1. Install dependencies:

```bash
pip install torch transformers gtts flask pyaudio numpy
```

2. Run the Flask API server:

```bash
python3 real_time_speech_processor.py
```

3. Open the frontend website:

Serve the `web` directory using a simple HTTP server, for example:

```bash
python3 -m http.server 8001 -d web
```

Then open your browser and navigate to `http://localhost:8001/index.html`.

## Usage

- Click the "Start Recording" button on the website to record your voice.
- The recorded audio will be sent to the backend for transcription.
- The transcription will be corrected and displayed.
- The corrected text will be converted to speech and played back.

## Notes

- The backend API runs on port 8000.
- The frontend is a static website using Tailwind CSS, Google Fonts, and Font Awesome.
- Ensure your microphone permissions are enabled in the browser.

## License

MIT License
