import os
from PIL import Image

class Dataset:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        import pandas as pd
        self.data = pd.read_csv(self.file_path)

    def preprocess(self):
        if self.data is not None:
            # Example preprocessing steps
            self.data.dropna(inplace=True)
            self.data = (self.data - self.data.mean()) / self.data.std()  # Standardization
        else:
            raise ValueError("Data not loaded. Please load the data first.")

    def load_images_from_folder(self, folder_path):
        images = []
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                image = Image.open(file_path)
                images.append((filename, image))
            except Exception as e:
                print(f"Error loading image {filename}: {e}")
        return images

    @staticmethod
    def save_image(image, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        image.save(file_path)