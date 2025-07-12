My First Project: "Small" Voice Activity Detection App

😏 What’s this all about?
So yeah, this is my first-ever project, and I didn’t want to start with "Hello World" or make a calculator like every next chomu out there.
Instead, I went full throttle into Voice Activity Detection (VAD) with some serious audio & video analysis and real-time speech-to-text (STT) magic.

And because looking good is half the battle won, I packed it all inside a sleek, dark-themed GUI using CustomTkinter. 😎
Whether you’re analyzing audio files, ripping speech out of videos, or just want your mic to act like Jarvis — this tool’s got your back.

🏗️ Project Structure
Here's how the machine works under the hood (don’t worry, no rats running inside):

  * UnifiedApp: The mastermind. Handles GUI + logic. Basically, cool stuff.

  * VADNet: My custom PyTorch neural net model that detects those "hmm, someone is talking!" moments.

  * Live STT: Plug in your mic, say something cool, and boom — it gets transcribed. Only if it passes the VAD test.

  * File Analyzer: Load audio/video files, visualize the sound, play-pause it, and extract audio like a hacker.

  * File STT: Feed in any audio file and let it spit out clean text using VOSK + KaldiRecognizer. Offline. Safe. Desi approved.


