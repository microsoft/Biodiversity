import os
import json

from .annotation_creator import AnnotationCreator


class BaseReader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.dataset_name = os.path.basename(data_path)
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

    def process_dataset(self):
        """Executes the full dataset processing pipeline."""
        self.add_dataset_info()
        self.add_sounds()
        self.add_categories()
        self.add_annotations()
        self.save_dataset()
        self.load_dataset()
