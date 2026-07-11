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
import socket
import ssl
import re
warnings.filterwarnings("ignore")
from rich.console import Console

# ================== VTS CONFIG ===========================
VTS_PORT = 8001
MOUTH_PARAM = "MouthOpen"  # VTuber mouth parameter
TOKEN_FILE = "vts_token.json"

# ================== TWITCH CONFIG =======================
TWITCH_ENABLED = False            # Set to True to enable Twitch chat
TWITCH_CHANNEL = "your_channel"   # Your Twitch channel name (lowercase)
TWITCH_OAUTH = "oauth:your_token" # Get from https://twitchapps.com/tmi/
TWITCH_BOT_NAME = "your_bot"      # Your bot's Twitch username (lowercase)
TWITCH_RESPOND_ALL = False        # True = respond to all messages, False = only !ask <msg>
TWITCH_COOLDOWN = 5               # Seconds between responses

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

# OBS Subtitles instance
from obs_subtitles import OBSSubtitles
subtitles = OBSSubtitles()

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
parser.add_argument("--voice-model", type=str, default="chatterbox-turbo", choices=["chatterbox", "chatterbox-turbo"], help="TTS model: chatterbox (quality) or chatterbox-turbo (speed)")
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
    def __init__(self, voice_path=None, cfg_weight=0.5, model_type="chatterbox-turbo"):
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
        self.available_hotkeys = []
        self.hotkeys_detected = False

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
                self.request_hotkeys()
            elif mtype == "HotkeysInCurrentModelResponse":
                self.available_hotkeys = msg["data"]["availableHotkeys"]
                self.hotkeys_detected = True
                self.auto_detect_hotkeys(self.available_hotkeys)
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

    def request_hotkeys(self):
        return self.send_request(
            "list_hotkeys",
            "HotkeysInCurrentModelRequest"
        )

    def auto_detect_hotkeys(self, hotkeys_list):
        keywords = {
            "happy": ["happy", "joy", "smile", "laugh", "excited", "cheer"],
            "sad": ["sad", "cry", "depress", "upset", "frown", "tear"],
            "angry": ["angry", "mad", "rage", "furious", "annoyed", "upset"],
            "thinking": ["think", "hmm", "idea", "confuse", "wonder", "curious"],
            "neutral": ["neutral", "calm", "default", "normal", "idle", "still"],
        }
        console.print("\n[magenta]=== Available Hotkeys ===[/magenta]")
        for hotkey in hotkeys_list:
            name = hotkey.get("name", "")
            hotkey_id = hotkey.get("hotkeyID", "")
            console.print(f"  {name} -> {hotkey_id}")
            hname = name.lower()
            for emotion, kws in keywords.items():
                if any(kw in hname for kw in kws):
                    if not EMOTION_HOTKEYS.get(emotion):
                        EMOTION_HOTKEYS[emotion] = hotkey_id
                        console.print(f"[green]    => Auto-mapped to '{emotion}'[/green]")
                    break
        console.print(f"[magenta]==========================[/magenta]\n")

# Initialize VTS client
vtube = VTubeStudioClient()

# ======================================================

# ================== TWITCH IRC CLIENT ==================

class TwitchChatListener:
    def __init__(self):
        self.host = "irc.chat.twitch.tv"
        self.port = 6697
        self.sock = None
        self.running = False
        self.thread = None
        self.message_queue = queue.Queue()

    def connect(self):
        try:
            self.sock = ssl.wrap_socket(socket.socket(socket.AF_INET, socket.SOCK_STREAM))
            self.sock.connect((self.host, self.port))
            self.sock.settimeout(1.0)
            self._send_raw(f"PASS {TWITCH_OAUTH}")
            self._send_raw(f"NICK {TWITCH_BOT_NAME}")
            self._send_raw(f"JOIN #{TWITCH_CHANNEL}")
            self.running = True
            self.thread = threading.Thread(target=self._listen_loop, daemon=True)
            self.thread.start()
            console.print(f"[green]Connected to Twitch: #{TWITCH_CHANNEL}[/green]")
            return True
        except Exception as e:
            console.print(f"[red]Twitch connection failed:[/red] {e}")
            return False

    def _send_raw(self, message):
        if self.sock:
            self.sock.send((message + "\r\n").encode("utf-8"))

    def send_chat(self, message):
        if self.sock and self.running:
            self._send_raw(f"PRIVMSG #{TWITCH_CHANNEL} :{message}")

    def _listen_loop(self):
        buffer = ""
        while self.running:
            try:
                data = self.sock.recv(4096).decode("utf-8", errors="ignore")
                if not data:
                    console.print("[yellow]Twitch connection lost. Reconnecting...[/yellow]")
                    self._reconnect()
                    continue
                buffer += data
                while "\r\n" in buffer:
                    line, buffer = buffer.split("\r\n", 1)
                    self._handle_line(line)
            except socket.timeout:
                continue
            except Exception as e:
                console.print(f"[red]Twitch listen error:[/red] {e}")
                self._reconnect()

    def _handle_line(self, line):
        if line.startswith("PING"):
            self._send_raw("PONG :tmi.twitch.tv")
            return
        if "PRIVMSG" not in line:
            return
        try:
            parts = line.split("!", 1)
            username = parts[0].lstrip(":")
            msg_start = line.split(":", 2)
            if len(msg_start) < 3:
                return
            message = msg_start[2]
            self.message_queue.put((username, message))
        except Exception:
            pass

    def _reconnect(self):
        self.running = False
        try:
            self.sock.close()
        except Exception:
            pass
        time.sleep(5)
        try:
            self.sock = ssl.wrap_socket(socket.socket(socket.AF_INET, socket.SOCK_STREAM))
            self.sock.connect((self.host, self.port))
            self.sock.settimeout(1.0)
            self._send_raw(f"PASS {TWITCH_OAUTH}")
            self._send_raw(f"NICK {TWITCH_BOT_NAME}")
            self._send_raw(f"JOIN #{TWITCH_CHANNEL}")
            self.running = True
            console.print("[green]Reconnected to Twitch![/green]")
        except Exception as e:
            console.print(f"[red]Twitch reconnect failed:[/red] {e}")
            self.running = False

    def get_messages(self):
        messages = []
        while not self.message_queue.empty():
            messages.append(self.message_queue.get())
        return messages

    def stop(self):
        self.running = False
        try:
            self.sock.close()
        except Exception:
            pass

# Initialize Twitch listener
twitch_chat = None
if TWITCH_ENABLED:
    twitch_chat = TwitchChatListener()

# ======================================================

class AIVtuber:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.is_listening = True
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.silence_threshold = 0.008
        self.silence_duration = 1.5
        self.min_speech_duration = 0.3
        self.min_audio_length = 0.8
        self.max_record_duration = 20.0
        self.noise_floor_multiplier = 1.8
        self.warmup_seconds = 0.8

        print("Initializing voice converter...")
        self.voice_converter = rvc_converter

        self.audio = pyaudio.PyAudio()
        self.stream = None
        self._last_twitch_response = 0.0

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

    def detect_speech(self, audio_buffer, noise_floor=None):
        rms = np.sqrt(np.mean(audio_buffer**2))
        threshold = max(self.silence_threshold, (noise_floor * self.noise_floor_multiplier) if noise_floor else self.silence_threshold)
        return rms > threshold, rms

    def flush_audio_queue(self):
        drained = 0
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                drained += 1
            except queue.Empty:
                break
        return drained

    def record_until_silence(self):
        drained = self.flush_audio_queue()
        if drained > 0:
            print(f"  (flushed {drained} stale chunks)")

        print("Waiting for speech...")
        audio_buffer = []
        silence_frames = 0
        noise_floor = None
        noise_samples = []
        warmup_frames = int(self.warmup_seconds * self.sample_rate / self.chunk_size)
        max_silence_frames = int(self.silence_duration * self.sample_rate / self.chunk_size)
        max_record_frames = int(self.max_record_duration * self.sample_rate / self.chunk_size)
        total_frames = 0
        started = False

        while self.is_listening and total_frames < max_record_frames:
            try:
                chunk = self.audio_queue.get(timeout=0.1)
                audio_buffer.append(chunk)
                total_frames += 1

                rms = np.sqrt(np.mean(chunk**2))

                if not started:
                    noise_samples.append(rms)
                    if total_frames >= warmup_frames and noise_floor is None:
                        noise_floor = np.percentile(noise_samples, 15)
                        print(f"  noise floor: {noise_floor:.6f}")

                if noise_floor is not None:
                    threshold = max(self.silence_threshold, noise_floor * self.noise_floor_multiplier)
                    is_speech = rms > threshold
                else:
                    is_speech = False

                if is_speech:
                    if not started:
                        started = True
                        print("Speech detected...")
                    silence_frames = 0
                elif started:
                    silence_frames += 1

                if started and silence_frames >= max_silence_frames:
                    break

            except queue.Empty:
                continue

        if not started:
            return None

        audio = np.concatenate(audio_buffer)

        start_idx = 0
        trailing = int(0.3 * self.sample_rate / self.chunk_size)
        for i in range(len(audio_buffer)):
            rms = np.sqrt(np.mean(audio_buffer[i]**2))
            threshold = max(self.silence_threshold, noise_floor * self.noise_floor_multiplier)
            if rms > threshold:
                start_idx = max(0, (i - trailing)) * self.chunk_size
                break

        audio = audio[start_idx:]
        return audio

    def transcribe(self, audio_data):
        duration = len(audio_data) / self.sample_rate
        if duration < self.min_audio_length:
            return ""

        print(f"Transcribing ({duration:.1f}s)...")
        audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-10)
        result = stt.transcribe(
            audio_data,
            fp16=WHISPER_FP16,
            temperature=0.0,
            condition_on_previous_text=False,
            no_speech_threshold=0.4,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            without_timestamps=True,
        )
        text = (result.get("text") or "").strip() if isinstance(result, dict) else ""
        word_count = len(text.split())
        if word_count <= 1:
            return ""
        return text

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

    def play_tts_with_mouth(self, audio, sr, subtitle_text=""):
        global tts_muted
        with audio_lock:
            block_size = 256
            alpha = 0.7
            prev_amp = 0.0

            # Mute microphone input during TTS playback to prevent feedback
            tts_muted = True
            vtube.set_mouth(0.0)

            if subtitle_text:
                subtitles.begin(subtitle_text)
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

                subtitles.update()

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
        self.play_tts_with_mouth(audio, sr, subtitle_text=text)
        subtitles.finish()

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

    def process_twitch_messages(self):
        if not twitch_chat:
            return

        now = time.time()
        if now - self._last_twitch_response < TWITCH_COOLDOWN:
            return

        messages = twitch_chat.get_messages()
        for username, message in messages:
            if not TWITCH_RESPOND_ALL and not message.lower().startswith("!ask "):
                continue

            user_text = message[5:] if message.lower().startswith("!ask ") else message
            user_text = user_text.strip()
            if not user_text:
                continue

            console.print(f"[cyan]Twitch [{username}]:[/cyan] {user_text}")
            ai_response = self.generate_response(f"Twitch user {username} says: {user_text}")
            twitch_chat.send_chat(ai_response)
            self._last_twitch_response = now

            self.speak(ai_response)
            emotion = self.detect_emotion(ai_response)
            self.trigger_animations(emotion)
            break

    def run(self):
        self.start_listening()

        if twitch_chat:
            twitch_chat.connect()

        try:
            while self.is_listening:
                self.process_twitch_messages()

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
            if twitch_chat:
                twitch_chat.stop()

    def detect_emotion(self, text):
        try:
            result = ollama.generate(
                model="llama3.2",
                prompt=f"Classify the emotion of this text. Reply with exactly one word: happy, sad, angry, thinking, or neutral.\n\nText: {text}\n\nEmotion:"
            )
            emotion = result["response"].strip().lower().rstrip(".,!")
            for e in ["happy", "sad", "angry", "thinking", "neutral"]:
                if e in emotion:
                    return e
        except Exception:
            pass
        return "neutral"

if __name__ == "__main__":
    vtuber = AIVtuber()
    
    console.print("[magenta]=== Animation Mappings ===[/magenta]")
    for emotion, hotkey_id in EMOTION_HOTKEYS.items():
        status = "Set" if hotkey_id else "Auto-detecting..."
        console.print(f"  {emotion}: {hotkey_id or '?'} ({status})")
    console.print("[magenta]==========================[/magenta]")
    if args.voice:
        console.print(f"[cyan]Voice cloning: {args.voice} ({args.voice_model}, cfg={args.cfg_weight})[/cyan]")
    else:
        console.print("[cyan]Voice: default built-in (use --voice <file.wav> for cloning)[/cyan]")

    if TWITCH_ENABLED:
        console.print(f"[green]Twitch: #{TWITCH_CHANNEL} as {TWITCH_BOT_NAME} (respond_all={TWITCH_RESPOND_ALL})[/green]")
    else:
        console.print("[dim]Twitch: disabled (set TWITCH_ENABLED = True to enable)[/dim]")
    
    vtuber.run()

