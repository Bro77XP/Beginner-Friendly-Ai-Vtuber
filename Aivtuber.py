import asyncio
import threading
import queue
import numpy as np
import pyaudio
import whisper
import ollama
import torch
import json
import os
import time
import argparse
import websocket
import sounddevice as sd
import soundfile as sf
from scipy.io import wavfile
import warnings
import math
warnings.filterwarnings("ignore")
from rich.console import Console

# ================== VTS CONFIG ===========================
VTS_PORT = 8001
MOUTH_PARAM = "MouthOpen"  # VTuber mouth parameter
TOKEN_FILE = "vts_token.json"

# Hotkey IDs for VTube Studio animations
EMOTION_HOTKEYS = {
    "happy": "",      # Auto-detect: names containing "happy", "joy", "smile"
    "sad": "",        # Auto-detect: names containing "sad", "cry", "depress"
    "angry": "",      # Auto-detect: names containing "angry", "mad", "upset"
    "thinking": "",   # Auto-detect: names containing "think", "hmm", "idea"
    "neutral": "",    # Auto-detect: names containing "neutral", "calm", "default"
}

console = Console()

# Audio lock to prevent conflicts between TTS and singing
audio_lock = threading.Lock()
tts_muted = False  # Flag to prevent feedback during TTS playback
# ======================================================

# Whisper configuration
WHISPER_MODEL = "base.en"
WHISPER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_FP16 = True if WHISPER_DEVICE == "cuda" else False

# Load Whisper model once at startup
console.print(f"Loading Whisper model '{WHISPER_MODEL}' on {WHISPER_DEVICE}...")
stt = whisper.load_model(WHISPER_MODEL, device=WHISPER_DEVICE)

# ------------------ CONSTANTS ----------------------
MAX_RECORD_SECONDS = 20
NO_WORD_TIMEOUT = 20
WORD_CHECK_INTERVAL = 4.0
MIN_CHECK_AUDIO_SEC = 1.5

# ------------------ ARGUMENTS ----------------------
parser = argparse.ArgumentParser()
parser.add_argument("--voice", type=str, help="Audio file (.wav/.mp3) for zero-shot voice cloning")
parser.add_argument("--voice-model", type=str, default="chatterbox", choices=["chatterbox", "chatterbox-turbo"], help="TTS model: chatterbox (quality) or chatterbox-turbo (speed)")
parser.add_argument("--cfg-weight", type=float, default=0.5, help="CFG weight for voice cloning (0.0-1.0, higher = closer to reference voice)")
parser.add_argument("--model", type=str, default="qwen2.5")
parser.add_argument("--whisper-model", type=str, default=WHISPER_MODEL, help="Whisper model to use (base, base.en, small, medium, large)")
parser.add_argument("--no-whisper-fp16", action="store_true", help="Disable fp16 for CPU inference")
args = parser.parse_args()

# Override config based on args if provided
if args.whisper_model:
    WHISPER_MODEL = args.whisper_model
if args.no_whisper_fp16:
    WHISPER_FP16 = False

# Reload Whisper model if args were provided
if args.whisper_model or args.no_whisper_fp16:
    console.print(f"Reloading Whisper model '{WHISPER_MODEL}' with fp16={WHISPER_FP16}...")
    stt = whisper.load_model(WHISPER_MODEL, device=WHISPER_DEVICE)

# ------------------ CHATTERBOX ZERO-SHOT VOICE CLONING ----------------------
class VoiceSynthesizer:
    def __init__(self, voice_path=None, cfg_weight=0.5, model_type="chatterbox"):
        self.voice_path = voice_path
        self.cfg_weight = cfg_weight
        self.model_type = model_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.load_model()

    def load_model(self):
        try:
            if self.model_type == "chatterbox-turbo":
                from chatterbox.tts_turbo import ChatterboxTurboTTS
                self.model = ChatterboxTurboTTS.from_pretrained(device=self.device)
                console.print("[green]Chatterbox Turbo TTS loaded (faster inference)[/green]")
            else:
                from chatterbox.tts import ChatterboxTTS
                self.model = ChatterboxTTS.from_pretrained(device=self.device)
                console.print("[green]Chatterbox TTS loaded (quality mode)[/green]")

            if self.voice_path and os.path.isfile(self.voice_path):
                console.print(f"[green]Zero-shot voice cloning enabled with: {self.voice_path}[/green]")
            elif self.voice_path:
                console.print(f"[yellow]Voice file not found: {self.voice_path}. Using default voice.[/yellow]")
                self.voice_path = None
            else:
                console.print("[yellow]No --voice specified. Using default built-in voice.[/yellow]")
        except Exception as e:
            console.print(f"[red]Failed to load Chatterbox TTS: {e}[/red]")

    def convert_voice(self, text):
        if not self.model:
            console.print("[red]Chatterbox TTS not loaded[/red]")
            sr = 24000
            return np.zeros(int(0.5 * sr), dtype=np.float32), sr

        try:
            kwargs = {
                "text": text,
                "cfg_weight": self.cfg_weight,
            }
            if self.voice_path and os.path.isfile(self.voice_path):
                kwargs["audio_prompt_path"] = self.voice_path
                console.print(f"[cyan]Generating speech (voice cloned from {os.path.basename(self.voice_path)})...[/cyan]")
            else:
                console.print("[cyan]Generating speech (default voice)...[/cyan]")

            wav = self.model.generate(**kwargs)
            return wav.squeeze().cpu().numpy(), self.model.sr
        except Exception as e:
            console.print(f"[red]Chatterbox TTS failed: {e}[/red]")
            sr = 24000
            return np.zeros(int(0.5 * sr), dtype=np.float32), sr


# Initialize voice synthesizer
rvc_converter = VoiceSynthesizer(
    voice_path=args.voice if args.voice else None,
    cfg_weight=args.cfg_weight,
    model_type=args.voice_model
)

# ------------------ VTS CLIENT ----------------------
class VTubeStudioClient:
    API_NAME = "VTubeStudioPublicAPI"
    API_VERSION = "1.0"

    def __init__(self):
        self.url = f"ws://localhost:{VTS_PORT}"

        self.ws = None
        self.thread = None

        self.authenticated = False
        self.token = self.load_token()

        self.hotkeys = {}

        self._last_mouth = -1.0
        self._last_mouth_time = 0.0

        self._connect()

        # Debug: auth state
        try:
            print("Authenticated:", self.authenticated)
        except Exception:
            pass

    def _connect(self):
        console.print(f"[cyan]Connecting to VTube Studio on {self.url}...[/cyan]")
        self.ws = websocket.WebSocketApp(
            self.url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        self.thread = threading.Thread(target=self.ws.run_forever, daemon=True)
        self.thread.start()

    def on_open(self, ws):
        console.print("[green]Connected to VTube Studio API[/green]")
        time.sleep(1)

        if self.token:
            console.print("[cyan]Attempting to authenticate with stored token...[/cyan]")
            self.send_request(
                "auth",
                "AuthenticationRequest",
                {
                    "pluginName": "Local AI VT3.0",
                    "pluginDeveloper": "Bro77xp",
                    "authenticationToken": self.token
                }
            )
        else:
            console.print("[cyan]No stored token found. Requesting authentication token...[/cyan]")
            self.send_request(
                "auth_token",
                "AuthenticationTokenRequest",
                {
                    "pluginName": "Local AI VT3.0",
                    "pluginDeveloper": "Bro77xp",
                    "pluginIcon": ""
                }
            )

    def on_message(self, ws, message):
        try:
            msg = json.loads(message)
            mtype = msg.get("messageType", "")
            if mtype == "AuthenticationTokenResponse":
                token = msg["data"]["authenticationToken"]
                self.save_token(token)
                console.print("[green]✅ Token received and saved. Authenticating...[/green]")
                self.send_request(
                    "auth",
                    "AuthenticationRequest",
                    {
                        "pluginName": "Local AI VT3.0",
                        "pluginDeveloper": "Bro77xp",
                        "authenticationToken": token
                    }
                )
            elif mtype == "AuthenticationResponse":
                self.authenticated = True
                console.print("[green]Authenticated with VTube Studio![/green]")
        except Exception as e:
            console.print(f"[red]Error parsing message:[/red] {e}")

    def on_error(self, ws, error):
        console.print(f"[red]VTS WebSocket error:[/red] {error}")

    def on_close(self, ws, close_status_code, close_msg):
        console.print("[yellow]VTS connection closed. Retrying in 5 seconds...[/yellow]")
        time.sleep(5)
        self._connect()

    def save_token(self, token):
        with open(TOKEN_FILE, "w") as f:
            json.dump({"token": token}, f)

    # -------------------------
    # Generic packet sender
    # -------------------------

    def send_request(
        self,
        request_id: str,
        message_type: str,
        data: dict | None = None
    ):
        if not self.is_connected():
            return False

        payload = {
            "apiName": self.API_NAME,
            "apiVersion": self.API_VERSION,
            "requestID": request_id,
            "messageType": message_type,
        }

        if data:
            payload["data"] = data

        try:
            self.ws.send(json.dumps(payload))
            return True
        except Exception as e:
            console.print(f"[red]VTS Send Error:[/red] {e}")
            return False

    def is_connected(self):
        return (
            self.ws
            and self.ws.sock
            and self.ws.sock.connected
        )

    # -------------------------
    # Mouth Control
    # -------------------------

    def set_mouth(self, value: float):
        """
        Optimized mouth injection.
        Skips tiny updates.
        """

        if not self.authenticated:
            return False

        value = max(0.0, min(1.0, float(value)))

        now = time.perf_counter()

        delta = abs(value - self._last_mouth)

        if (
            delta < 0.02
            and now - self._last_mouth_time < 0.02
        ):
            return False

        self._last_mouth = value
        self._last_mouth_time = now

        return self.send_request(
            "mouth",
            "InjectParameterDataRequest",
            {
                "parameterValues": [
                    {
                        "id": MOUTH_PARAM,
                        "value": value
                    }
                ]
            }
        )

    def close_mouth(self):
        self.set_mouth(0.0)

    def load_token(self):
        try:
            if not os.path.exists(TOKEN_FILE):
                return None

            with open(TOKEN_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

            return data.get("token") or data.get("authenticationToken")
        except Exception as e:
            console.print(f"[yellow]Could not load token: {e}[/yellow]")
            return None

    # -------------------------
    # Hotkeys
    # -------------------------

    def trigger_hotkey(self, hotkey_id: str):
        return self.send_request(
            f"hotkey_{hotkey_id}",
            "HotkeyTriggerRequest",
            {
                "hotkeyID": hotkey_id
            }
        )

# Initialize VTS client
vtube = VTubeStudioClient()

# ======================================================

class AIVtuber:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.is_listening = True
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.silence_threshold = 0.01
        self.silence_duration = 1.5
        self.min_speech_duration = 0.5

        print("Loading Whisper model...")
        self.whisper_model = whisper.load_model("base")

        print("Initializing voice converter...")
        self.voice_converter = rvc_converter

        self.audio = pyaudio.PyAudio()
        self.stream = None

    def audio_callback(self, in_data, frame_count, time_info, status):
        audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        self.audio_queue.put(audio_data)
        return (in_data, pyaudio.paContinue)

    def start_listening(self):
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self.audio_callback
        )
        self.stream.start_stream()
        print("Listening... (Press Ctrl+C to stop)")

    def stop_listening(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()

    def detect_speech(self, audio_buffer):
        rms = np.sqrt(np.mean(audio_buffer**2))
        return rms > self.silence_threshold

    def record_until_silence(self):
        print("Waiting for speech...")
        audio_buffer = []
        silence_frames = 0
        speech_frames = 0
        max_silence_frames = int(self.silence_duration * self.sample_rate / self.chunk_size)
        min_speech_frames = int(self.min_speech_duration * self.sample_rate / self.chunk_size)

        while self.is_listening:
            try:
                chunk = self.audio_queue.get(timeout=0.1)
                audio_buffer.append(chunk)

                is_speech = self.detect_speech(chunk)

                if is_speech:
                    speech_frames += 1
                    silence_frames = 0
                else:
                    silence_frames += 1

                if speech_frames >= min_speech_frames and silence_frames >= max_silence_frames:
                    break

            except queue.Empty:
                continue

        return np.concatenate(audio_buffer)

    def transcribe(self, audio_data):
        print("Transcribing...")
        result = self.whisper_model.transcribe(audio_data, fp16=torch.cuda.is_available())
        text = result.get("text") if isinstance(result, dict) else None
        return (text or "").strip()

    def generate_response(self, user_text):
        print(f"User: {user_text}")
        print("Generating response...")

        response = ollama.chat(
            model="llama3.2",
            messages=[
                {"role": "system", "content": "You are a lil baka Annoy the user as much as humanily possible as a ai vtuber Gaymer but keep it around 20 words or less."},
                {"role": "user", "content": user_text}
            ]
        )

        ai_response = response["message"]["content"]
        print(f"VTuber: {ai_response}")
        return ai_response

    def play_tts_with_mouth(self, audio, sr):
        global tts_muted
        with audio_lock:
            block_size = 256
            alpha = 0.7
            prev_amp = 0.0

            # Mute microphone input during TTS playback to prevent feedback
            tts_muted = True
            vtube.set_mouth(0.0)

            sd.play(audio, sr)
            start_time = time.time()
            num_blocks = len(audio) // block_size

            for i in range(num_blocks):
                block = audio[i * block_size:(i + 1) * block_size]
                rms = np.sqrt(np.mean(block ** 2))
                amp = float(np.clip(rms * 4.0, 0.0, 1.0))
                smoothed_amp = prev_amp * (1 - alpha) + amp * alpha
                vtube.set_mouth(smoothed_amp)
                prev_amp = smoothed_amp

                expected_time = start_time + (i * block_size / sr)
                sleep_time = expected_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)

            vtube.set_mouth(0.0)
            sd.wait()

            # Unmute microphone input after TTS playback
            tts_muted = False

    def speak(self, text):
        audio, sr = self.voice_converter.convert_voice(text)
        self.play_tts_with_mouth(audio, sr)

    def trigger_animations(self, emotion):
        if not vtube.authenticated:
            console.print("[yellow]VTS not authenticated, skipping animations[/yellow]")
            return

        console.print(f"[magenta]Triggering {emotion} animation...[/magenta]")

        if emotion == "happy" and EMOTION_HOTKEYS["happy"]:
            vtube.trigger_hotkey(EMOTION_HOTKEYS["happy"])
        elif emotion == "sad" and EMOTION_HOTKEYS["sad"]:
            vtube.trigger_hotkey(EMOTION_HOTKEYS["sad"])
        elif emotion == "angry" and EMOTION_HOTKEYS["angry"]:
            vtube.trigger_hotkey(EMOTION_HOTKEYS["angry"])
        elif emotion == "thinking" and EMOTION_HOTKEYS["thinking"]:
            vtube.trigger_hotkey(EMOTION_HOTKEYS["thinking"])
        elif emotion == "neutral" and EMOTION_HOTKEYS["neutral"]:
            vtube.trigger_hotkey(EMOTION_HOTKEYS["neutral"])


    def run(self):
        self.start_listening()

        try:
            while self.is_listening:
                audio_data = self.record_until_silence()

                if audio_data is not None and len(audio_data) > 0:
                    user_text = self.transcribe(audio_data)

                    if user_text:
                        ai_response = self.generate_response(user_text)
                        self.speak(ai_response)

                        emotion = self.detect_emotion(ai_response)
                        self.trigger_animations(emotion)

        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.stop_listening()

    def detect_emotion(self, text):
        t = text.lower()
        if any(w in t for w in ["angry", "mad", "hate", "shut up"]):
            return "angry"
        if any(w in t for w in ["happy", "love", "fun", "yay"]):
            return "happy"
        if any(w in t for w in ["sad", "sorry", "lonely"]):
            return "sad"
        if any(w in t for w in ["think", "hmm", "maybe"]):
            return "thinking"
        return "neutral"

if __name__ == "__main__":
    vtuber = AIVtuber()
    
    console.print("[magenta]=== Animation Mappings ===[/magenta]")
    for emotion, hotkey_id in EMOTION_HOTKEYS.items():
        status = "Auto-detected" if not hotkey_id else "Manual" if hotkey_id else "Not configured"
        console.print(f"  {emotion}: {hotkey_id} ({status})")
    console.print("[magenta]==========================[/magenta]")
    if args.voice:
        console.print(f"[cyan]Voice cloning: {args.voice} ({args.voice_model}, cfg={args.cfg_weight})[/cyan]")
    else:
        console.print("[cyan]Voice: default built-in (use --voice <file.wav> for cloning)[/cyan]")
    
    vtuber.run()

