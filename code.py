import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QWidget, QTabWidget, QFileDialog
from PyQt5.QtCore import QTimer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pygame
import parselmouth
import numpy as np
import seaborn as sns

sns.set()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voice Analysis and Playback")
        self.setGeometry(100, 100, 800, 600)
        self.initUI()

        # Initialize variables for real-time updates
        self.audio_path = None  # No default audio file
        self.snd = None  # Parselmouth Sound object
        self.current_time = 0.0
        self.time_step = 0.1  # Increased to 100 ms for better analysis
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.is_paused = False

    def initUI(self):
        # Set up a tabbed layout
        tabs = QTabWidget()
        self.setCentralWidget(tabs)

        # Add tabs
        self.analysis_tab = QWidget()
        tabs.addTab(self.analysis_tab, "Audio & Voice Analysis")

        # Set up the Analysis tab layout
        self.analysis_layout = QVBoxLayout()
        self.analysis_tab.setLayout(self.analysis_layout)

        # Add a Matplotlib canvas
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.analysis_layout.addWidget(self.canvas)

        # Add buttons
        load_button = QPushButton("Load Audio File")
        load_button.clicked.connect(self.load_audio_file)
        self.analysis_layout.addWidget(load_button)

        play_button = QPushButton("Play Audio and Real-Time Plot")
        play_button.clicked.connect(self.play_audio_and_plot)
        self.analysis_layout.addWidget(play_button)

        pause_button = QPushButton("Pause Audio and Real-Time Plot")
        pause_button.clicked.connect(self.pause_audio_and_plot)
        self.analysis_layout.addWidget(pause_button)

        stop_button = QPushButton("Stop Audio and Real-Time Plot")
        stop_button.clicked.connect(self.stop_audio_and_plot)
        self.analysis_layout.addWidget(stop_button)

    def load_audio_file(self):
        """Open a file dialog to select an audio file."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Audio File", "", "Audio Files (*.wav *.mp3)")
        if file_path:
            self.audio_path = file_path
            try:
                # Load the new audio file
                self.snd = parselmouth.Sound(self.audio_path)
                if pygame.mixer.get_init():
                    pygame.mixer.music.load(self.audio_path)  # Update pygame with the new file
                print(f"Loaded audio file: {self.audio_path}")
                self.current_time = 0.0  # Reset the current time
            except Exception as e:
                print(f"Error loading audio file: {e}")

    def play_audio_and_plot(self):
        """Play audio and start or resume real-time plot updates."""
        try:
            if not self.audio_path or not self.snd:
                print("No audio file loaded. Please load an audio file first.")
                return

            if not pygame.mixer.get_init():
                pygame.mixer.init()
                pygame.mixer.music.load(self.audio_path)  # Ensure pygame has the correct file loaded

            if self.is_paused:
                pygame.mixer.music.unpause()
                self.timer.start(int(self.time_step * 1000))
                self.is_paused = False
            else:
                pygame.mixer.music.play()
                self.current_time = 0.0
                self.timer.start(int(self.time_step * 1000))

        except Exception as e:
            print(f"Error during playback: {e}")

    def pause_audio_and_plot(self):
        """Pause both audio playback and real-time plotting."""
        if pygame.mixer.get_init():
            pygame.mixer.music.pause()
        self.timer.stop()
        self.is_paused = True

    def stop_audio_and_plot(self):
        """Completely stop audio playback and real-time plotting."""
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
        self.timer.stop()
        self.is_paused = False
        self.current_time = 0.0
        self.figure.clear()
        self.canvas.draw()

    def update_plot(self):
        """Update the plot in real-time."""
        try:
            if not pygame.mixer.music.get_busy():
                self.timer.stop()
                return

            end_time = self.current_time + self.time_step
            snd_chunk = self.snd.extract_part(from_time=self.current_time, to_time=end_time)

            # Check if the audio chunk is valid
            if snd_chunk.get_total_duration() <= 0:
                print("Audio chunk is empty or invalid.")
                self.timer.stop()
                return

            pitch = snd_chunk.to_pitch()
            intensity = snd_chunk.to_intensity()
            spectrogram = snd_chunk.to_spectrogram()

            self.figure.clear()

            ax1 = self.figure.add_subplot(311)
            self.draw_spectrogram(ax1, spectrogram)

            ax2 = self.figure.add_subplot(312)
            self.plot_pitch(ax2, pitch)

            ax3 = self.figure.add_subplot(313)
            self.plot_intensity(ax3, intensity)

            self.canvas.draw()
            self.current_time += self.time_step

        except Exception as e:
            print(f"Error during real-time plotting: {e}")

    def draw_spectrogram(self, ax, spectrogram, dynamic_range=70):
        X, Y = spectrogram.x_grid(), spectrogram.y_grid()
        sg_db = 10 * np.log10(spectrogram.values)
        mesh = ax.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range, cmap='afmhot')
        ax.set_title("Spectrogram")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [Hz]")
        plt.colorbar(mesh, ax=ax, label="Power (dB)")

    def plot_pitch(self, ax, pitch):
        ax.plot(pitch.xs(), pitch.selected_array['frequency'], label="Pitch (Hz)")
        ax.set_title("Pitch Analysis")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [Hz]")
        ax.legend()
        ax.grid(True)

    def plot_intensity(self, ax, intensity):
        """Plot the intensity (dB) of the audio."""
        if intensity is not None:
            try:
                times = intensity.xs()
                values = intensity.values.flatten()

                # Remove NaN values
                valid_indices = ~np.isnan(values)
                times = times[valid_indices]
                values = values[valid_indices]

                if len(times) > 0 and len(values) > 0:
                    ax.plot(times, values, color='orange', label="Intensity (dB)")
                    ax.set_title("Intensity Analysis")
                    ax.set_xlabel("Time [s]")
                    ax.set_ylabel("Intensity [dB]")
                    ax.legend()
                    ax.grid(True)
                else:
                    print("No valid intensity data to plot.")
                    ax.set_title("Intensity Analysis (No valid data)")
                    ax.set_xlabel("Time [s]")
                    ax.set_ylabel("Intensity [dB]")
                    ax.grid(True)
            except Exception as e:
                print(f"Error in plotting intensity: {e}")
        else:
            print("Intensity data is None.")
            ax.set_title("Intensity Analysis (No data)")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Intensity [dB]")
            ax.grid(True)

# Run the application
app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())
