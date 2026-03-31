from datetime import datetime
from typing import Optional
import soundfile as sf
import pandas as pd
import requests
import json
import os

class AnnotationCreator:
    """
    A class to create and manage bioacustics annotations in a standard format.

    Attributes:
        data (dict): A dictionary to store general info, sounds, and annotations of a bioacustics dataset.
    """

    def __init__(self):
        """
        Initializes the AnnotationCreator with an empty dataset.
        """
        self.data = {
            "info": {},
            "categories": [],
            "sounds": [],
            "annotations": []
        }

    def _validate_date_format(self, date_str: str, date_format: str = "%Y%m%d"):
        """
        Validates if the date string matches the given format and is not a future date.

        Args:
            date_str (str): The date string to validate.
            date_format (str): The expected format of the date string.

        Raises:
            ValueError: If the date string does not match the format or is a future date.
        """
        try:
            date_obj = datetime.strptime(date_str, date_format)
        except ValueError:
            raise ValueError(f"Date '{date_str}' is not in the correct format. Expected format: {date_format}")

        if date_obj > datetime.now():
            raise ValueError("Future dates are invalid.")

    def _get_duration_and_sample_rate(self, file_path:str):
        """
        Get the duration and sample rate of a sound file

        Args:
            file_path (str): The path to the sound file

        Returns:
            duration (float): The duration of the sound file in seconds
            sample_rate (int): The sample rate of the sound file in Hz
        """
        with sf.SoundFile(file_path) as sound_file:
            duration = len(sound_file) / sound_file.samplerate
            sample_rate = sound_file.samplerate
        return duration, sample_rate

    def add_info(self, title:str=None,  license:str=None, publication_date:Optional[datetime]=None, description:Optional[str]=None, creators:Optional[list]=None, version:Optional[float]=None, url:Optional[str]=None):
        """
        Adds the general information about the dataset.

        Args:
            title (str): The title of the bioacoustic dataset.
            license (str): The name of the license that specifies the permissions and restrictions for using the bioacoustic dataset.
            publication_date (Optional[datetime]): The date when the bioacoustic dataset was published in YYYY-MM-DD format.
            description (Optional[str]): A brief summary of the dataset.
            creators (Optional[dict]): List with creators information.
            version (Optional[float]): The version number of the dataset.
            url (Optional[str]): The web address where the dataset can be accessed or downloaded. If it's a Zenodo URL, metadata is fetched automatically.

        Raises:
            ValueError: If the year is in the future or the date format is incorrect.
        """
        if url is not None and "zenodo.org/records/" in url:
            try:
                record_id = url.split("zenodo.org/records/")[1]
                response = requests.get(f"https://zenodo.org/api/records/{record_id}")
                data = response.json()
                metadata = data['metadata']
                title=metadata['title']
                license=metadata['license']['id']
                publication_date=datetime.strptime(metadata['publication_date'], "%Y-%m-%d").strftime("%Y%m%d")
                description=metadata['description']
                creators=metadata['creators']
                if 'version' in metadata:
                    version=metadata['version']
                else:
                    version=metadata['relations']['version']

            except requests.exceptions.RequestException as e:
                raise ValueError(f"Failed to fetch metadata from Zenodo: {e}")

        if publication_date:
            self._validate_date_format(publication_date)

        self.data["info"]["title"] = title
        self.data["info"]["license"] = license
        self.data["info"]["publication_date"] = publication_date
        self.data["info"]["description"] = description
        self.data["info"]["creators"] = creators
        self.data["info"]["version"] = version
        self.data["info"]["url"] = url

    def add_categories(self, categories_df:pd.DataFrame):
        """
        Adds the categories ids and names to the dataset.

        Args:
            categories (DataFrame): A pandas DataFrame containing the category information.
        """
        sorted_df = categories_df.sort_values(by=categories_df.columns[0])
        sorted_df.reset_index(drop=True, inplace=True)
        sorted_df.index.name = 'id'
        categories_list = sorted_df.reset_index().to_dict(orient='records')
        self.data['categories'] = categories_list

    def add_sound(self, id:int, file_name_path:str, duration:int, sample_rate:int, latitude:float, longitude:float, date_recorded:Optional[datetime]=None, project:Optional[str]=None):
        """
        Adds a sound entry to the dataset.

        Args:
            id (int): A unique identifier for a specific sound within the dataset.
            file_name_path (str): The path of the audio file containing the bioacoustic recording.
            duration (float): The length of the audio recording in seconds.
            sample_rate (int): The number of samples of audio carried per second, measured in Hz.
            latitude (float): The geographical latitude where the bioacoustic recording was taken.
            longitude (float):The geographical longitude where the bioacoustic recording was taken.
            date_recorded (Optional[str]): The datetime when the audio was recoded in YYYY-MM-DD format.
            project (Optional[str]): The name of the project associated with the sound recording.
        Raises:
            ValueError: If duration, sample rate, latitude, or longitude are out of valid range,
                        or if the date format is incorrect.
        """
        if duration <= 0:
            raise ValueError("Duration must be a positive value.")
        if sample_rate <= 0:
            raise ValueError("Sample rate must be a positive value.")
        if latitude is not None and not (isinstance(latitude, float) and pd.isna(latitude)):
            if not (-90 <= latitude <= 90):
                raise ValueError("Latitude must be between -90 and 90 degrees.")
        if longitude is not None and not (isinstance(longitude, float) and pd.isna(longitude)):
            if not (-180 <= longitude <= 180):
                raise ValueError("Longitude must be between -180 and 180 degrees.")
        if date_recorded:
            self._validate_date_format(date_recorded)

        sound = {
            "id": id,
            "file_name_path": file_name_path,
            "duration": duration,
            "sample_rate": sample_rate,
            "latitude": latitude,
            "longitude": longitude,
            "date_recorded": date_recorded,
            "project": project,
        }
        self.data['sounds'].append(sound)

    def add_annotation(
        self,
        anno_id:int,
        sound_id:int,
        category_id:int,
        category:str,
        t_min:float,
        t_max:float,
        supercategory:Optional[str]=None,
        f_min:Optional[float]=None,
        f_max:Optional[float]=None,
        ):
        """
        Adds an annotation entry to the dataset.

        Args:
            anno_id (int):  A unique identifier for a specific annotation within the dataset.
            sound_id (int): The identifier of the sound to which the annotation is linked.
            category_id (int): The identifier of the category to which the sound belongs.
            category (str): The name of the category of sounds, such as a particular species or type of call.
            t_min (float): The starting time of the annotated sound within the recording, in seconds.
            t_max (float): The ending time of the annotated sound within the recording, in seconds.
            supercategory (Optional[str]): A higher-level grouping that the category belongs to.
            f_min (Optional[float]): The lowest frequency of the annotated sound within the recording, in Hz.
            f_max (Optional[float]): The highest frequency of the annotated sound within the recording, in Hz
        Raises:
            ValueError: If any of the provided values are out of valid range, or if the
                        time/frequency constraints are violated.

        """
        sound_dict = self.data['sounds'][sound_id]
        if t_min < 0:
            raise ValueError("t_min must be a positive value.")
        if t_max < 0:
            raise ValueError("t_max must be a positive value.")
        if t_max > sound_dict["duration"]:
            t_max = sound_dict["duration"]
        if t_max < t_min:
            raise ValueError(
                f"t_max ({t_max:.4f}) must be greater than t_min ({t_min:.4f}). "
                f"Sound duration is {sound_dict['duration']:.4f}s "
                f"(anno_id={anno_id}, sound_id={sound_id})."
            )
        if f_min and f_max:
            if f_min < 0:
                raise ValueError("f_min must be a positive value.")
            if f_max < 0:
                raise ValueError("f_max must be a positive value.")
            if f_max < f_min:
                raise ValueError("f_max must be greater than f_min.")
            if f_max > sound_dict["sample_rate"] / 2:
                raise ValueError("f_max must be less than half the sample rate of the sound.")

        annotation = {
            "anno_id": anno_id,
            "sound_id": sound_id,
            "category_id": category_id,
            "category": category,
            "supercategory": supercategory,
            "t_min": t_min,
            "t_max": t_max,
            "f_min": f_min,
            "f_max": f_max,
        }
        self.data['annotations'].append(annotation)

    def convert_crowsetta_bbox_annotations(self, crowsetta_annotations:list):
        """
        Adds annotations from Crowsetta to the dataset.

        Args:
            crowsetta_annotations (list): A list of annotations in Crowsetta format.
        """
        # Add categories
        unique_labels = set()
        for annot in crowsetta_annotations:
            for bbox in annot.bboxes:
                unique_labels.add(bbox.label)

        categories_df = pd.DataFrame(list(unique_labels), columns=['label'])
        self.add_categories(categories_df)

        # Add sounds
        for sound_id, annotation in enumerate(crowsetta_annotations):
            duration, sample_rate = self._get_duration_and_sample_rate(annotation.notated_path)
            self.add_sound(id=sound_id,
                           file_name_path=annotation.notated_path.name,
                           duration=duration,
                           sample_rate=sample_rate,
                           latitude=None,
                           longitude=None)
            # Add annotations
            for anno_id, bbox in enumerate(annotation.bboxes):
                category_id = [category for category in self.data["categories"] if category["label"] == bbox.label][0]["id"]
                self.add_annotation(anno_id=anno_id,
                                    sound_id=sound_id,
                                    category_id=category_id,
                                    category=bbox.label,
                                    t_min=float(bbox.onset),
                                    t_max=float(bbox.offset),
                                    f_min=float(bbox.low_freq),
                                    f_max=float(bbox.high_freq))

    def convert_crowsetta_seq_annotations(self, crowsetta_annotations:list):
        """
        Adds annotations from Crowsetta to the dataset.

        Args:
            crowsetta_annotations (list): A list of annotations in Crowsetta format.
        """
        # Add categories
        unique_labels = set()
        for annot in crowsetta_annotations:
            for segment in annot.seq.segments:
                unique_labels.add(segment.label)

        categories_df = pd.DataFrame(list(unique_labels), columns=['label'])
        self.add_categories(categories_df)

        # Add sounds
        for sound_id, annotation in enumerate(crowsetta_annotations):
            duration, sample_rate = self._get_duration_and_sample_rate(annotation.notated_path)
            self.add_sound(id=sound_id,
                           file_name_path=annotation.notated_path.name,
                           duration=duration,
                           sample_rate=sample_rate,
                           latitude=None,
                           longitude=None)
            # Add annotations
            for anno_id, segment in enumerate(annotation.seq.segments):
                category_id = [category for category in self.data["categories"] if category["label"] == segment.label][0]["id"]
                self.add_annotation(anno_id=anno_id,
                                    sound_id=sound_id,
                                    category_id=category_id,
                                    category=segment.label,
                                    t_min=float(segment.onset_s),
                                    t_max=float(segment.offset_s))

    def save_to_file(self, filename):
        """
        Saves the current dataset to a JSON file.

        Args:
            filename (str): The name of the file to save the dataset to.
        """
        with open(filename, 'w') as f:
            json.dump(self.data, f, indent=4)

class BaseReader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.output_path = os.path.join(data_path, "annotations.json")
        self.annotation_creator = AnnotationCreator()
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
    
    def show_summary(self):
        """Displays a general summary of the dataset."""
        total_duration = sum(sound['duration'] for sound in self.sounds)
        total_hours = total_duration / 3600

        print(f"Dataset: {self.data['info']['title']}")
        print(f"Total categories: {len(self.categories)}")
        print(f"Total audio recordings: {len(self.sounds)}")
        print(f"Total annotations: {len(self.annotations)}")
        print(f"Total duration: {total_hours:.2f} hours")

    def process_dataset(self):
        """Executes the full dataset processing pipeline."""
        self.add_dataset_info()
        self.add_sounds()
        self.add_categories()
        self.add_annotations()
        self.save_dataset()
        self.load_dataset()
        self.show_summary()
