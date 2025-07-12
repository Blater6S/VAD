
import os, sys, threading, queue, json, tempfile
import numpy as np
import torch
import torch.nn as nn
import sounddevice as sd
import soundfile as sf
import librosa, librosa.display
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog, messagebox
import customtkinter as ctk
from pydub import AudioSegment
from moviepy import VideoFileClip
from vosk import Model, KaldiRecognizer

# ─── Constants ─────────────────────────────────────────
AUDIO_EXT = ('.wav', '.mp3', '.flac', '.ogg')
VIDEO_EXT = ('.mp4', '.mkv', '.mov', '.avi')
SAMPLE_RATE = 16000
FRAME_DURATION = 1.0
MEL_BINS = 40
HOP_SIZE = 0.010
FRAME_SIZE = 0.025
THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "vad_simple.pth"

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

VOSK_MODEL_PATH = resource_path("vosk-model-small-en-us-0.15")

# ─── VAD Model ─────────────────────────────────────────
class VADNet(nn.Module):
    def __init__(self, input_dim=MEL_BINS, hidden_dim=64):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x, _ = self.rnn(x)
        return self.classifier(x).squeeze(-1)

# ─── Main GUI App ──────────────────────────────────────
class UnifiedApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("🎧 Unified Audio/Video Analyzer + VAD + STT")
        self.geometry("1000x700")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.audio_data = None
        self.sample_rate = None
        self.current_file = None
        self.is_playing = False
        self.running = False
        self.q = queue.Queue()

        self.vad_model = VADNet().to(DEVICE)
        self.vad_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        self.vad_model.eval()

        self.vosk_model = Model(VOSK_MODEL_PATH)
        self.vosk_recognizer = KaldiRecognizer(self.vosk_model, SAMPLE_RATE)
        self.vosk_recognizer.SetWords(True)

        self.init_gui()

    def init_gui(self):
        tabview = ctk.CTkTabview(self)
        tabview.pack(fill="both", expand=True, padx=10, pady=10)

        self.tab_analyzer = tabview.add("🔍 Analyzer")
        self.tab_live = tabview.add("🎤 Live VAD+STT")
        self.tab_file = tabview.add("📂 File Transcriber")

        self.build_analyzer_tab()
        self.build_live_tab()
        self.build_file_tab()

    # ─── ANALYZER TAB ──────────────────────────────────
    def build_analyzer_tab(self):
        control_frame = ctk.CTkFrame(self.tab_analyzer)
        control_frame.pack(fill="x", padx=5, pady=5)

        ctk.CTkButton(control_frame, text="Select File", command=self.select_file).pack(side="left", padx=5)
        ctk.CTkButton(control_frame, text="Extract Audio (Video)", command=self.extract_audio_dialog).pack(side="left", padx=5)
        ctk.CTkButton(control_frame, text="🗂️ Extract From Multiple Videos", command=self.extract_multiple_videos).pack(side="left", padx=5)
        ctk.CTkButton(control_frame, text="▶️ Play", command=self.play_audio).pack(side="left", padx=5)
        ctk.CTkButton(control_frame, text="⏹ Stop", command=self.stop_audio).pack(side="left", padx=5)

        self.spectrogram_var = ctk.IntVar(value=0)
        ctk.CTkCheckBox(control_frame, text="Spectrogram", variable=self.spectrogram_var, command=self.update_plot).pack(side="right")

        self.fig_analyzer, (self.ax_wave, self.ax_zoom) = plt.subplots(2, 1, figsize=(8, 5), dpi=100)
        self.fig_analyzer.subplots_adjust(hspace=0.5)
        self.canvas_analyzer = FigureCanvasTkAgg(self.fig_analyzer, master=self.tab_analyzer)
        self.canvas_analyzer.get_tk_widget().pack(fill="both", expand=True)

    def extract_audio_from_video(self, video_path, output_path):
        try:
            video = VideoFileClip(video_path)
            audio = video.audio
            temp_wav = tempfile.mktemp(suffix=".wav")
            audio.write_audiofile(temp_wav, fps=16000, nbytes=2, buffersize=2000, codec='pcm_s16le', ffmpeg_params=["-ac", "1"])
            seg = AudioSegment.from_wav(temp_wav).set_channels(1).set_frame_rate(16000)
            seg.export(output_path, format='wav')
            os.remove(temp_wav)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to extract from {video_path}:\n{e}")

    def extract_audio_dialog(self):
        if not self.current_file or not self.current_file.lower().endswith(VIDEO_EXT):
            messagebox.showerror("Error", "Please select a valid video file.")
            return
        save_path = filedialog.asksaveasfilename(defaultextension=".wav")
        if save_path:
            self.extract_audio_from_video(self.current_file, save_path)
            messagebox.showinfo("Success", f"Extracted to:\n{save_path}")

    def extract_multiple_videos(self):
        video_paths = filedialog.askopenfilenames(title="Select Video Files", filetypes=[("Video files", "*.mp4 *.mkv *.avi *.mov")])
        if not video_paths:
            return
        for video in video_paths:
            base = os.path.splitext(os.path.basename(video))[0]
            default_name = base + "_extracted.wav"
            save_path = filedialog.asksaveasfilename(title=f"Save audio for {base}", defaultextension=".wav", initialfile=default_name)
            if save_path:
                self.extract_audio_from_video(video, save_path)

    def select_file(self):
        path = filedialog.askopenfilename(filetypes=[("Media", "*.wav *.mp3 *.flac *.ogg *.mp4 *.mkv *.avi *.mov")])
        if path:
            self.current_file = path
            self.load_audio(path)

    def load_audio(self, path):
        try:
            if path.lower().endswith(VIDEO_EXT):
                temp_wav = tempfile.mktemp(suffix=".wav")
                self.extract_audio_from_video(path, temp_wav)
                y, sr = sf.read(temp_wav)
                os.remove(temp_wav)
            else:
                y, sr = sf.read(path)

            if y.ndim > 1:
                y = y.mean(axis=1)
            if y.dtype != np.float32:
                y = y.astype(np.float32) / (np.iinfo(y.dtype).max if np.issubdtype(y.dtype, np.integer) else 1)

            self.audio_data = y
            self.sample_rate = sr
            self.update_plot()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def update_plot(self):
        if self.audio_data is None:
            return
        self.ax_wave.clear()
        self.ax_zoom.clear()
        duration = len(self.audio_data) / self.sample_rate
        times = np.linspace(0, duration, num=len(self.audio_data))
        db = 20 * np.log10(np.abs(self.audio_data) + 1e-10)
        loudest_idx = np.argmax(db)
        loudest_time = times[loudest_idx]
        self.ax_wave.plot(times, self.audio_data, color='cyan')
        self.ax_wave.axvline(loudest_time, color='red', linestyle='--', label=f"Loudest: {db[loudest_idx]:.2f} dB")
        self.ax_wave.legend()
        self.ax_wave.set_title("Waveform")
        if self.spectrogram_var.get():
            S = librosa.stft(self.audio_data, n_fft=1024, hop_length=256)
            S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
            img = librosa.display.specshow(S_db, sr=self.sample_rate, x_axis='time', y_axis='hz', ax=self.ax_zoom, cmap='magma')
            self.fig_analyzer.colorbar(img, ax=self.ax_zoom)
            self.ax_zoom.set_title("Spectrogram")
        else:
            zoom_sec = 0.5
            start = max(0, loudest_time - zoom_sec / 2)
            end = min(duration, loudest_time + zoom_sec / 2)
            idx_start = int(start * self.sample_rate)
            idx_end = int(end * self.sample_rate)
            zoom_times = np.linspace(start, end, idx_end - idx_start)
            self.ax_zoom.plot(zoom_times, self.audio_data[idx_start:idx_end], color='orange')
            self.ax_zoom.set_title("Zoomed Loudest Section")
        self.canvas_analyzer.draw()

    def play_audio(self):
        if self.audio_data is None or self.is_playing:
            return
        def _play():
            self.is_playing = True
            try:
                sd.play(self.audio_data, self.sample_rate)
                sd.wait()
            except Exception as e:
                messagebox.showerror("Playback", str(e))
            self.is_playing = False
        threading.Thread(target=_play, daemon=True).start()

    def stop_audio(self):
        if self.is_playing:
            sd.stop()
            self.is_playing = False

    # ─── LIVE VAD TAB ──────────────────────────────────
    def build_live_tab(self):
        ctk.CTkButton(self.tab_live, text="🎙️ Start Live", command=self.start_live).pack(pady=10)
        self.live_text = ctk.CTkTextbox(self.tab_live, height=100)
        self.live_text.pack(padx=10, fill="x")
        self.fig_live, self.ax_live = plt.subplots(figsize=(8, 2))
        self.line_live, = self.ax_live.plot([], [], color='cyan')
        self.ax_live.set_ylim(-1, 1)
        self.ax_live.set_xlim(0, int(SAMPLE_RATE * FRAME_DURATION))
        self.canvas_live = FigureCanvasTkAgg(self.fig_live, master=self.tab_live)
        self.canvas_live.get_tk_widget().pack()

    def start_live(self):
        if self.running:
            return
        self.running = True
        threading.Thread(target=self.live_loop, daemon=True).start()

    def live_loop(self):
        try:
            with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=8000, dtype='int16', channels=1,
                                   callback=lambda indata, frames, time, status: self.q.put(bytes(indata))):
                while self.running:
                    data = self.q.get()
                    audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    mel = librosa.feature.melspectrogram(y=audio_np.flatten(), sr=SAMPLE_RATE,
                                                         n_fft=int(FRAME_SIZE*SAMPLE_RATE),
                                                         hop_length=int(HOP_SIZE*SAMPLE_RATE),
                                                         n_mels=MEL_BINS)
                    log_mel = librosa.power_to_db(mel).T
                    input_tensor = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        prob = self.vad_model(input_tensor).squeeze(0).cpu().numpy().mean()
                    self.line_live.set_ydata(audio_np)
                    self.line_live.set_xdata(np.arange(len(audio_np)))
                    self.ax_live.set_xlim(0, len(audio_np))
                    self.canvas_live.draw()
                    if prob > THRESHOLD and self.vosk_recognizer.AcceptWaveform(data):
                        result = json.loads(self.vosk_recognizer.Result())
                        text = result.get("text", "")
                        if text:
                            self.live_text.insert("end", text + "\n")
                            self.live_text.see("end")
        except Exception as e:
            print("Live error:", e)

    # ─── FILE STT TAB ─────────────────────────────────
    def build_file_tab(self):
        ctk.CTkButton(self.tab_file, text="📂 Select Audio", command=self.transcribe_audio_file).pack(pady=10)
        self.file_text = ctk.CTkTextbox(self.tab_file, height=200)
        self.file_text.pack(padx=10, fill="both", expand=True)

    def transcribe_audio_file(self):
        path = filedialog.askopenfilename(filetypes=[("Audio", "*.wav *.mp3")])
        if not path:
            return
        y, sr = librosa.load(path, sr=SAMPLE_RATE)
        recognizer = KaldiRecognizer(self.vosk_model, sr)
        recognizer.SetWords(True)
        chunk_size = int(SAMPLE_RATE * FRAME_DURATION)
        for i in range(0, len(y), chunk_size):
            chunk = (y[i:i+chunk_size] * 32768).astype(np.int16).tobytes()
            if recognizer.AcceptWaveform(chunk):
                result = json.loads(recognizer.Result())
                text = result.get("text", "")
                if text:
                    self.file_text.insert("end", text + "\n")
                    self.file_text.see("end")

if __name__ == "__main__":
    app = UnifiedApp()
    app.mainloop()
