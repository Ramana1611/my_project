import pyaudio
import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, pipeline
from gtts import gTTS
import os
import tempfile
from flask import Flask, request, jsonify, send_file
import io
import logging
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealTimeSpeechProcessor:
    def __init__(self):
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Speech Recognition (Wav2Vec 2.0)
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(self.device)
        
        # Text Correction (GPT-2)
        self.llm = pipeline(
            "text-generation", 
            model="gpt2",
            device=0 if str(self.device) == "cuda" else -1
        )

        # Audio Config
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.RECORD_SECONDS = 5

    def capture_audio(self, duration=5):
        """Capture audio with PyAudio."""
        if duration <= 0:
            raise ValueError("Duration must be positive")
            
        p = pyaudio.PyAudio()
        stream = None
        frames = []
        
        try:
            stream = p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )
            
            logger.info(f"Recording for {duration} seconds...")
            for _ in range(0, int(self.RATE / self.CHUNK * duration)):
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                frames.append(np.frombuffer(data, dtype=np.int16))
            
            audio = np.concatenate(frames)
            return audio, self.RATE
            
        except Exception as e:
            logger.error(f"Audio capture error: {str(e)}")
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
        
        try:
            inputs = self.processor(
                audio, 
                sampling_rate=self.RATE, 
                return_tensors="pt", 
                padding=True
            ).input_values.to(self.device)
            
            with torch.no_grad():
                logits = self.asr_model(inputs).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = self.processor.batch_decode(predicted_ids)[0]
                return transcription.lower().strip()
        except Exception as e:
            logger.error(f"Speech to text error: {str(e)}")
            return ""

    def correct_text(self, text):
        """Correct text using GPT-2."""
        if not text.strip():
            return ""
        
        prompt = f"Please correct this text: {text}\nCorrected version:"
        try:
            corrected = self.llm(
                prompt, 
                max_length=100, 
                num_return_sequences=1,
                temperature=0.7
            )[0]['generated_text']
            
            # Extract the corrected portion
            corrected_text = corrected.split("Corrected version:")[-1].strip()
            return corrected_text
        except Exception as e:
            logger.error(f"Text correction error: {str(e)}")
            return text

    def text_to_speech(self, text):
        """Convert text to speech using gTTS."""
        if not text.strip():
            return None
        
        try:
            tts = gTTS(text=text, lang='en', slow=False)
            audio_bytes = io.BytesIO()
            tts.write_to_fp(audio_bytes)
            audio_bytes.seek(0)
            return audio_bytes
        except Exception as e:
            logger.error(f"Text to speech error: {str(e)}")
            return None

app = Flask(__name__)
processor = RealTimeSpeechProcessor()

@app.route('/api/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        audio_bytes = audio_file.read()
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
        transcription = processor.speech_to_text(audio_np)
        return jsonify({'transcription': transcription})
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        return jsonify({'error': 'Processing failed'}), 500

@app.route('/api/correct', methods=['POST'])
def correct():
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
    
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    corrected_text = processor.correct_text(data['text'])
    return jsonify({'corrected_text': corrected_text})
