import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import librosa
import librosa.display
import argparse
import matplotlib.ticker as ticker
from collections import Counter

from .annotation_creator import AnnotationCreator


class BaseReader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.dataset_name = os.path.basename(data_path)
        self.output_path = os.path.join(data_path, "annotations.json")
        self.annotation_creator = AnnotationCreator()
        self.visualization_dir = os.path.join(data_path, "visualizations")
        self.data = None

    def add_dataset_info(self):
        """Method to add dataset metadata (to be implemented in subclasses)."""
        raise NotImplementedError("This method should be implemented in a subclass.")

    def add_sounds(self):
        """Method to add sounds (to be implemented in subclasses)."""
        raise NotImplementedError("This method should be implemented in a subclass.")

    def add_categories(self):
        """Method to add categories (to be implemented in subclasses)."""
        raise NotImplementedError("This method should be implemented in a subclass.")

    def add_annotations(self):
        """Method to add annotations (to be implemented in subclasses)."""
        raise NotImplementedError("This method should be implemented in a subclass.")

    def save_dataset(self):
        """Saves the processed dataset as a JSON file."""
        self.annotation_creator.save_to_file(self.output_path)

    def load_dataset(self):
        """Loads the dataset from the JSON file."""
        with open(self.output_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.categories = {cat["id"]: cat["name"] for cat in self.data["categories"]}
        self.sounds = self.data["sounds"]
        self.annotations = self.data["annotations"]

        if not os.path.exists(self.visualization_dir):
            os.makedirs(self.visualization_dir)

    def visualizations(self):
        """Generates visualizations for the dataset."""

        def show_summary(self):
            """Displays a general summary of the dataset."""
            total_duration = sum(sound['duration'] for sound in self.sounds)
            total_hours = total_duration / 3600

            print(f"Dataset: {self.dataset_name}")
            print(f"Total species: {len(self.categories)}")
            print(f"Total audio recordings: {len(self.sounds)}")
            print(f"Total annotations: {len(self.annotations)}")
            print(f"Total duration: {total_hours:.2f} hours")

        def get_category_distribution(self):
            """Displays the class distribution in the dataset and saves it as an image."""
            category_counts = Counter([anno['category'] for anno in self.annotations])
            sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)

            plt.figure(figsize=(12, 6))
            sns.barplot(x=[cat[0] for cat in sorted_categories], y=[cat[1] for cat in sorted_categories])
            plt.xticks(rotation=90, ha='right', fontsize=8)
            plt.xlabel("Species")
            plt.ylabel("Frequency")
            plt.title("Species distribution in the dataset")

            output_path = os.path.join(self.visualization_dir, "category_distribution.png")
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()

        def plot_duration_distribution(self):
            """Displays the distribution of audio durations and saves it as an image."""
            durations = [sound['duration'] for sound in self.sounds]

            plt.figure(figsize=(8, 5))
            sns.histplot(durations, bins=20, kde=True)
            plt.xlabel("Duration (s)")
            plt.ylabel("Frequency")
            plt.title("Audio duration distribution")

            output_path = os.path.join(self.visualization_dir, "duration_distribution.png")
            plt.savefig(output_path)
            plt.close()

        def plot_annotations_per_audio(self):
            """Displays the number of annotations per audio file and saves it as an image."""
            annotation_counts = Counter([anno['sound_id'] for anno in self.annotations])

            plt.figure(figsize=(12, 6))
            sns.barplot(x=list(annotation_counts.keys()), y=list(annotation_counts.values()))
            plt.xticks(rotation=90, ha='right', fontsize=8)
            plt.xlabel("Audio ID")
            plt.ylabel("Number of Annotations")
            plt.title("Annotations per Audio File")

            output_path = os.path.join(self.visualization_dir, "annotations_per_audio.png")
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()

        # TODO: Add a function to plot spectograms for sequence like annotations
        def plot_spectrogram_bbox(self, sound_id):
            """Generates a spectrogram for 5 annotations and saves it."""
            sound = next((s for s in self.sounds if s['id'] == sound_id), None)
            if sound is None:
                print(f"Audio ID {sound_id} not found.")
                return

            relevant_annotations = sorted(
                [anno for anno in self.annotations if anno['sound_id'] == sound_id],
                key=lambda x: x['t_min']
            )

            if not relevant_annotations:
                print(f"No annotations found for sound_id {sound_id}")
                return

            relevant_annotations = relevant_annotations[0:5]  # Limit to 5 annotations

            audio_path = os.path.join(self.data_path, sound['file_name_path'])
            y, sr = librosa.load(audio_path, sr=sound['sample_rate'])

            t_min_global = min(anno['t_min'] for anno in relevant_annotations)
            t_max_global = max(anno['t_max'] for anno in relevant_annotations)
            # Añadir un margen para mejor visualización (5 segundos antes y después)
            margin = 1
            t_min_display = max(0, t_min_global - margin)
            t_max_display = min(len(y) / sr, t_max_global + margin)

            # Extraer el segmento relevante del audio
            segment_start_sample = int(t_min_display * sr)
            segment_end_sample = int(t_max_display * sr)
            y_segment = y[segment_start_sample:segment_end_sample]

            # Calcular el espectrograma
            n_fft = 2048
            hop_length = 512
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y_segment, n_fft=n_fft, hop_length=hop_length)), ref=np.max)

            # Crear la figura
            plt.figure(figsize=(15, 8))

            # Mostrar el espectrograma
            img = librosa.display.specshow(
                D,
                sr=sr,
                hop_length=hop_length,
                x_axis='time',
                y_axis='log',
                cmap='magma',
                x_coords=np.linspace(t_min_display, t_max_display, D.shape[1])
            )

            # Ajustar los límites del eje Y (frecuencia)
            f_min_global = min(anno['f_min'] for anno in relevant_annotations)
            f_max_global = max(anno['f_max'] for anno in relevant_annotations)

            # Añadir margen en las frecuencias
            f_margin_percentage = 0.1
            f_margin_min = f_min_global * f_margin_percentage
            f_margin_max = f_max_global * f_margin_percentage

            plt.ylim([max(20, f_min_global - f_margin_min), f_max_global + f_margin_max])

            for i, anno in enumerate(relevant_annotations):
                print(f"Annotation {i}: t_min={anno['t_min']}, t_max={anno['t_max']}, f_min={anno['f_min']}, f_max={anno['f_max']}")

            # Personalizar el formato del eje X para mostrar minutos:segundos
            def format_time(x, pos=None):
                #minutes = int(x // 60)
                #seconds = int(x % 60)
                seconds = float(x)
                return f"{seconds}"

            ax = plt.gca()
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_time))

            # Dibujar las cajas de anotaciones
            for i, anno in enumerate(relevant_annotations):
                # Usamos directamente los tiempos absolutos
                t_min = anno['t_min']
                t_max = anno['t_max']

                # Crear el rectángulo (x, y, ancho, alto)
                width = t_max - t_min
                height = anno['f_max'] - anno['f_min']

                # Dibujar el rectángulo
                rect = plt.Rectangle(
                    (t_min, anno['f_min']),
                    width,
                    height,
                    linewidth=2,
                    edgecolor='red',
                    facecolor='none',
                    alpha=0.8
                )
                plt.gca().add_patch(rect)

                # Añadir etiqueta con el ID y categoría
                plt.text(
                    t_min + width/2,
                    anno['f_max'] + height*0.1,
                    f"{anno['category']} (ID: {anno['anno_id']})",
                    color='white',
                    fontsize=9,
                    ha='center',
                    va='bottom',
                    bbox=dict(facecolor='black', alpha=0.7)
                )

            # Configurar ejes y títulos
            plt.colorbar(img, format='%+2.0f dB')
            plt.title(f"Spectrogram of {os.path.basename(sound['file_name_path'])}", fontsize=14)
            plt.xlabel('Time (s)', fontsize=12)
            plt.ylabel('Frecuency (Hz)', fontsize=12)

            # Añadir anotación de tiempo absoluto
            plt.figtext(
                0.01, 0.01,
                f"Absolut time: {t_min_display:.2f}s - {t_max_display:.2f}s",
                fontsize=10
            )

            # Guardar la figura
            output_path = os.path.join(self.visualization_dir, f"spectrogram_sound_id_{sound_id}.png")
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.tight_layout()
            plt.show()
            plt.close()

        show_summary(self)
        # get_category_distribution(self)
        # plot_duration_distribution(self)
        # plot_annotations_per_audio(self)
        # if self.annotations[0]["f_min"] != None and self.annotations[0]["f_max"] != None:
        #     plot_spectrogram_bbox(self, sound_id=0)

    def process_dataset(self):
        """Executes the full dataset processing pipeline."""
        self.add_dataset_info()
        self.add_sounds()
        self.add_categories()
        self.add_annotations()
        self.save_dataset()
        self.load_dataset()
        self.visualizations()
