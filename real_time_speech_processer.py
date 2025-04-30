import pyaudio
import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, BertTokenizer, BertModel, pipeline
from gtts import gTTS
import os
import tempfile
from flask import Flask, request, jsonify, send_file
import io

class RealTimeSpeechProcessor:
    def __init__(self):
        # Initialize device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Speech Recognition (Wav2Vec 2.0)
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(self.device)
        
        # Text Correction (GPT-2)
        self.llm = pipeline("text-generation", model="gpt2", device=0 if self.device == "cuda" else -1)

        # Audio Config
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000

    def capture_audio(self, duration=5):
        """Capture audio with PyAudio."""
        p = pyaudio.PyAudio()
        stream = None
        frames = []
        
        try:
            # Open stream
            stream = p.open(format=self.FORMAT,
                            channels=self.CHANNELS,
                            rate=self.RATE,
                            input=True,
                            frames_per_buffer=self.CHUNK)
            
            print(f"\nRecording for {duration} seconds...")
            for _ in range(0, int(self.RATE / self.CHUNK * duration)):
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                frames.append(np.frombuffer(data, dtype=np.int16))
            
            audio = np.concatenate(frames)
            return audio, self.RATE
            
        except Exception as e:
            print(f"Audio error: {str(e)}")
            return None, self.RATE
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            p.terminate()

    def speech_to_text(self, audio):
        """Convert speech to text using Wav2Vec2."""
        if audio is None or len(audio) == 0:
            return ""
        
        inputs = self.processor(audio, sampling_rate=self.RATE, return_tensors="pt", padding=True).input_values.to(self.device)
        with torch.no_grad():
            logits = self.asr_model(inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]
            return transcription.lower().strip()

    def correct_text(self, text):
        """Correct text using GPT-2."""
        if not text.strip():
            return ""
        
        prompt = f"Please correct the following transcription for grammar and clarity:\n\"{text}\"\nCorrected version:"
        try:
            corrected = self.llm(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
            corrected_text = corrected.split("Corrected version:")[-1].strip()
            return corrected_text
        except Exception as e:
            print(f"LLM error: {str(e)}")
            return text

    def text_to_speech(self, text):
        """Convert text to speech using gTTS."""
        if not text.strip():
            return None
        
        try:
            tts = gTTS(text=text, lang='en')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                tts.save(fp.name)
                fp.seek(0)
                audio_bytes = fp.read()
            os.unlink(fp.name)
            return audio_bytes
        except Exception as e:
            print(f"TTS error: {str(e)}")
            return None

app = Flask(__name__)
processor = RealTimeSpeechProcessor()

@app.route('/api/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    audio_file = request.files['audio']
    audio_bytes = audio_file.read()
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
    transcription = processor.speech_to_text(audio_np)
    return jsonify({'transcription': transcription})

@app.route('/api/correct', methods=['POST'])
def correct():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    corrected_text = processor.correct_text(data['text'])
    return jsonify({'corrected_text': corrected_text})

@app.route('/api/speak', methods=['POST'])
def speak():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    audio_bytes = processor.text_to_speech(data['text'])
    if audio_bytes is None:
        return jsonify({'error': 'Text to speech failed'}), 500
    return send_file(io.BytesIO(audio_bytes), mimetype='audio/mpeg')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
