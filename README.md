# VAD
My first Project

this is just Small Voice Activity Detection Project

This project is a Python-based desktop application designed for audio and video analysis, voice activity detection (VAD), and speech-to-text transcription (STT) using deep learning and signal processing techniques. The GUI is built using CustomTkinter, and the application is intended for users who want a unified tool for working with audio from both files and live sources.


Project Structure
UnifiedApp: Main class for GUI and logic
VADNet: Neural network class for VAD
Live STT: Real-time microphone transcription with VAD gating
File Analyzer: Audio/video file waveform, spectrogram, and audio extraction
File STT: Offline transcription from files.

Features
📂 File Analyzer Tab
Load audio or video files and visualize:
Waveform
Zoomed loudest section
Spectrogram (optional toggle)

Extract audio from:
A single video file
Multiple video files in batch
Play and stop audio within the app
Identify and mark loudest moments in the waveform
Support for common formats: .wav, .mp3, .flac, .ogg, .mp4, .mkv, .avi, .mov

🎤 Live VAD + STT Tab
Perform real-time voice activity detection using a bi-directional LSTM model
Display live waveform visualization
Live speech-to-text transcription using VOSK
Only transcribes speech segments that pass the VAD threshold for noise reduction( if your microphone work properly)

🗣️ File Transcriber Tab
Load pre-recorded audio files for offline transcription
Segment audio into chunks and transcribe using VOSK + KaldiRecognizer
Display transcribed text in real-time inside the app

Module Used
PyTorch – for VAD deep learning model (VADNet)
Librosa – for audio processing and spectrogram computation
MoviePy & Pydub – for video and audio handling
VOSK – for speech recognition
Sounddevice – for live microphone streaming
Matplotlib – for plotting waveforms and spectrograms
CustomTkinter – for the modern, dark-themed GUI
Tkinter – for file dialogs and app layout
Multithreading – for smooth playback, live input, and UI responsiveness






