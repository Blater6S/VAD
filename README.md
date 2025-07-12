My First Project: “Small” Voice Activity Detection App

What’s This?
Not your average Hello World. I started off bold — with a voice activity detector, live speech-to-text, and audio-video analysis. All packed in a dark-themed CustomTkinter GUI.

🏗️ Project Structure

   * UnifiedApp: The brain. GUI + logic.
   * VADNet: My custom PyTorch model for speech detection.
   * Live STT: Real-time mic input + transcription.
   * File Analyzer: Load media, visualize audio, extract sounds.
   * File STT: Offline audio transcription via VOSK.

⚙️ Key Features
📂 File Analyzer Tab

  * Load audio/video (.mp3, .mp4, .wav, etc.)
  * View waveform, zoom loudest spots, toggle spectrogram
  * Extract audio (single or batch), play/pause in-app
  * Auto-mark loudest moments

🎤 Live VAD + STT Tab

  * Real-time BiLSTM-based speech detection
  * Transcribes only when actual voice detected
  * Live waveform + speech-to-text
  * Filters out fan and background noise
  * Advised (If your mic doesn’t work… well, that’s a you problem not a code problem )

🗣️ File Transcriber Tab

  * Load pre-recorded audio
  * Segments + transcribes in real-time
  * View transcript instantly

Module Stack

  * PyTorch, Librosa, MoviePy, Pydub, VOSK, Sounddevice, Matplotlib, CustomTkinter, Tkinter, Multithreading.
  * Offline. No cloud. No nonsense.

Why I Built This?
 Because existing tools suck. I wanted something offline, smart, fast, and clean. So I made it myself.

Bonus Vibes
Open app → load file → see sexy waveform dance.
Talk into mic → it types.
Batch transcribe → done.
Feels like Jarvis. Works like a beast.

My Final Thought
Don’t let the name fool you — it’s called “small,” but it hits like a rocket. Occasionally explodes too... but that’s part of the fun. 😎
( remember its just small project for fun(not accurate work yet ), unless you want to expand its capabilities for heavy work.)
