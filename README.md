# AI VTuber For Begginers/non programmers Easy To setup

An AI VTuber that uses Whisper for speech recognition, Ollama for LLM inference, and Chatterbox TTS in a continuous listening loop.

This Was Also Made On a AMD gpu But the code is mainly supported For cpu users So it can be used without amd or nvdia gpus

This uses Python 3.10.11 if you don't have it as your main Version do:
py -3.10 -m venv venv

(you can check the version with python -V)


(here's a video Tutorial on how to setup the ai vtuber https://youtu.be/b2Mr-ZUVqzo?si=ZUGchmHMS0Wb5Zv)

## Features

- **Whisper** (base.en model) - Real-time speech-to-text in English
- **Ollama** (llama3.2) - AI model for generating VTuber responses
- **Chatterbox TTS** - Text-to-speech to speak responses
- **Automatic silence detection** - Only records when speech is detected
- **Continuous listening loop** - Runs forever until Ctrl+C
- **VTube Studio integration** - Controls mouth expressions via VTube Studio Api

## Dependencies
(IMPORTANT!!!)
MAKE A VENV FIRST AND MAKE SURE YOU ARE INSIDE THE PROJECT FOLDER
for example

C:\Users\(Yourusername)\Downloads\Begginerfriendlyai

And right click and do "open in terminal"
----

or do cd C:\Users\(Yourusername)\Downloads\Begginerfriendlyai

THEN DO

python -m venv venv

then

venv\Scripts\Activate

or

Now Once your inside your virtual Environment (venv) do this

```bash
pip install -r requirements.txt
```

**Required external tools:**
- [Ollama](https://ollama.ai/) - Install and run: `ollama serve`
- [VTube Studio] - For character animation control

### Core Dependencies (Required - Works on Windows)
```bash
pip install openai-whisper ollama chatterbox-tts pyaudio numpy torch sounddevice soundfile websocket-client rich
```
### Optional RVC Voice Cloning (Advanced - Windows Build Required)
```bash
# Uncomment in requirements.txt or install manually (requires C++ build tools)
# pip install torchaudio librosa onnxruntime onnx fairseq pyworld praat-parselmouth TTS edge-tts
```

**Note:** RVC voice cloning is optional. The VTuber works perfectly with just the core dependencies using Chatterbox TTS. RVC voice cloning requires C++ build tools (Visual Studio Build Tools) and can be challenging to install on Windows.

## Quick Start

MAKE A VENV FIRST AND MAKE SURE YOU ARE INSIDE THE PROJECT FOLDER
for example

C:\Users\(Yourusername)\Downloads\Begginerfriendlyai

And right click and do "open in terminal"
----

or do cd C:\Users\(Yourusername)\Downloads\Begginerfriendlyai

THEN DO

python -m venv venv

then

venv\Scripts\Activate

or

Now Once your inside your virtual Environment (venv) do this


1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Pull models:
```bash
ollama pull llama3.2
python -m pip install openai-whisper
```

3. Start Ollama:
```bash
ollama serve
```

4. Run the VTuber:
```bash
python Aivtuber.py
```

## Configuration

The AI VTuber integrates with VTube Studio to control character animations:

### VTube Studio Integration

The script automatically connects to VTube Studio (port 8001) to control:

- **Mouth expressions**: Real-time mouth movement synchronized with speech
- **Emotion expressions**: Triggers pre-configured emotion hotkeys (happy, sad, angry, thinking, neutral)

**Setup instructions:**
1. Install VTube Studio and start it
2. Open the plugin "Local AI VTuber" from the VTube Studio plugins menu
3. The plugin will automatically generate an authentication token if one doesn't exist
4. Restart the AI VTuber script after initial setup
5. Configure VTube Studio mouth parameter:
   - Input: `MouthOpen`
   - Output: `ParamMouthOpenY`

<img width="603" height="551" alt="Screenshot 2026-06-24 214030" src="https://github.com/user-attachments/assets/63a9591e-bdb6-4928-bcb9-c79ffe403a32" />
### Configuring Animation Hotkeys

**Important:** You must configure the actual hotkey IDs in VTube Studio for animations to work:

1. In VTube Studio, create animations for each emotion:
   - Happy animation (e.g., "happy", "joyful", "smile")
   - Sad animation (e.g., "sad", "cry", "depressed")
   - Angry animation (e.g., "angry", "mad", "upset")
   - Thinking animation (e.g., "think", "hmm", "idea")
   - Neutral animation (e.g., "neutral", "calm", "default")

2. Set hotkey IDs for these animations in the VTube Studio plugin settings

3. Edit `Aivtuber.py` and update `EMOTION_HOTKEYS` with the actual hotkey IDs:

```python
EMOTION_HOTKEYS = {
    "happy": "your_happy_hotkey_id",    # Replace with actual hotkey ID
    "sad": "your_sad_hotkey_id",        # Replace with actual hotkey ID  
    "angry": "your_angry_hotkey_id",    # Replace with actual hotkey ID
    "thinking": "your_thinking_hotkey_id", # Replace with actual hotkey ID
    "neutral": "your_neutral_hotkey_id", # Replace with actual hotkey ID
}
```

4. **Auto-configuration option:** The script can auto-detect hotkeys based on name patterns. Leave empty to use auto-detection.

### Model Configuration

- **Whisper**: Uses "base.en" model for faster English speech recognition
- **Ollama**: Uses "llama3.2" model for AI responses
- **Chatterbox**: Automatically loads on startup

### Voice Configuration

Change the voice used by the VTuber:

```bash
# Use pre-trained Chatterbox voices
python Aivtuber.py --voice af_heart      # Female heart voice
python Aivtuber.py --voice am_sleepy     # Male sleepy voice
python Aivtuber.py --voice af_smiling    # Female smiling voice
python Aivtuber.py  # Uses default voice

# Use custom RVC voice model (RECOMMENDED for voice cloning)
python Aivtuber.py --voice ./my_rvc_model
python Aivtuber.py --voice /home/user/my_rvc_model
python Aivtuber.py --voice "C:\\Users\\l-ota\\Downloads\\Recording159.wav"
python Aivtuber.py --voice "C:\\Users\\l-ota\\OneDrive\\Documents\\Sound recordings\\Recording159.wav"
```

### Voice Options

**Pre-trained Voices (Chatterbox)**
- `af_heart` - Female heart voice
- `am_sleepy` - Male sleepy voice  
- `af_smiling` - Female smiling voice
- *(More voices available in Chatterbox)*

**Custom RVC Voice Models**
- Provide a directory path to an RVC voice model
- Directory must contain `infer.py` and model files
- Supports both Windows and Unix paths

### RVC Voice Setup Instructions
1. Download an RVC voice model from https://github.com/RVC-SFT/SVS
2. Extract the model to a directory
3. The directory should contain:
   - `infer.py` - RVC inference script
   - Model files (e.g., `model.pth`, `config.json`)
   - Other required files
4. Run the VTuber with the directory path

### Voice Parameter Behavior
- If a file path (with `.wav`, `.mp3` extension or `/` or `\` in path): Loads as custom voice
- If a directory path without extension: Tries to load as RVC model
- If empty or not provided: Uses default voice (Chatterbox)
- Invalid paths fall back to default voice with warning

### Advanced Features

1. **Automatic emotion detection**: The AI analyzes response text and detects emotions to trigger appropriate VTS hotkeys
2. **Response formatting**: The AI is prompted to be a cute anime VTuber with expressive responses
3. **Robust recording**: Advanced silence detection prevents unnecessary recording

## Usage

The AI VTuber runs in a continuous loop:

1. **Listening phase**: Waits for speech with automatic silence detection
2. **Speech detection**: Only starts recording after minimum speech duration is confirmed
3. **Transcribing**: Uses Whisper to convert speech to text
4. **AI response**: Ollama generates a VTuber-appropriate response
5. **Speaking**: Chatterbox TTS speaks the response aloud
6. **Mouth control**: VTube Studio controls mouth expressions in sync with speech
7. **Repeat**: Returns to listening mode

## Notes

- Press Ctrl+C to stop the VTuber at any time
- Ensure proper audio device permissions for microphone access
- For GPU acceleration, install PyTorch CUDA versions
- Adjust `silence_threshold`, `silence_duration`, and `min_speech_duration` in the code for different environments

## Troubleshooting

### Common Issues

1. **"Ollama not running" error**:
   - Make sure Ollama is installed and running with `ollama serve`
   - Verify the model "llama3.2" is pulled

2. **VTube Studio connection failed**:
   - Ensure VTube Studio is running
   - Check that VTS_PORT (default: 8001) is correct
   - Make sure VTube Studio plugins are enabled

3. **Audio permissions**:
   - Grant microphone permissions to this application
   - On Linux: `pip install pyaudio` might require additional system packages

4. **Model loading issues**:
   - Whisper uses "base.en" for faster performance
   - Ensure all dependencies are installed from requirements.txt

## Customization

### Adjusting Silence Detection

Edit `Aivtuber.py` and modify these constants:

```python
silence_threshold = 0.01    # Lower = more sensitive, Higher = less sensitive
silence_duration = 1.5      # Seconds of silence before stopping recording
min_speech_duration = 0.5   # Minimum speech duration to trigger recording
```

### Changing Models

- **Whisper**: Change in line 24: `self.whisper_model = whisper.load_model("base")`
- **Ollama**: Change in line 109: `model="llama3.2"`

### Adding Emotions

Edit the `EMOTION_HOTKEYS` dictionary in the code and add hotkeys to VTube Studio:

```python
EMOTION_HOTKEYS = {
    "happy": "your_happy_hotkey_id",
    "sad": "your_sad_hotkey_id",
    "angry": "your_angry_hotkey_id",
    "thinking": "your_thinking_hotkey_id",
    "neutral": "your_neutral_hotkey_id",
}
```

## Future Enhancements

Potential future improvements:

1. **Local LLM alternatives**: Support for other Ollama models or local LLM implementations
2. **Multi-language support**: Whisper language switching and response localization
3. **Context memory**: Maintain conversation history for more coherent interactions
4. **Advanced emotion system**: More nuanced emotion detection and expression control
5. **Stream processing**: WebSocket streaming for lower latency
6. **Plugin architecture**: Easy addition of new features and integrations


##questions

Is this Ali?

No this is Not ali In Fact Ali is A WAY more complicated program than this.
<img width="1142" height="43" alt="Screenshot 2026-06-24 220422" src="https://github.com/user-attachments/assets/3ee9f637-60d5-41d9-8702-61cb6875bb3f" />

This Also Doesn't use Any of ali's og code aside from How the mouth api works And Some recreated stuff Like the api being used so you can play music without having issues

Does This contain Any preMade vtuber models i can Download?

No But i do Have Older vtuber models you can use for this for example:

https://drive.google.com/file/d/1WGdSQxnzKeirUBSKeif4En2-b8AW7yTN/view?usp=sharing (IF you Don't want to use Hyori's model and test a different one)

Is the ai Sentient And planning arson 

-Proby not Unless You Replaced Ollama with something Else


## License

This project is open source. Feel free to modify and distribute as long as you give appropriate credit since that's really important to get a habit out of.
