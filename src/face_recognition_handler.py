import face_recognition
import os
import cv2
import numpy as np
from PIL import Image

class FaceRecognition:
    def __init__(self, known_faces_dir=None):
        self.known_faces = {}
        if known_faces_dir:
            self.known_faces = self.load_known_faces(known_faces_dir)

    def load_known_faces(self, folder_path):
        known_faces = {}
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            image = face_recognition.load_image_file(file_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_faces[filename] = encodings[0]
        return known_faces

    def recognize_faces(self, image):
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        results = []
        for encoding in face_encodings:
            matches = face_recognition.compare_faces(list(self.known_faces.values()), encoding)
            if True in matches:
                match_index = matches.index(True)
                person_name = list(self.known_faces.keys())[match_index]
                results.append(person_name)
            else:
                results.append("Unknown")
        return results

    def get_face_embedding(self, image):
        """
        Extracts the face embedding (encoding) for the first face detected in the image.
        Returns None if no face is detected.
        """
        # Convert PIL.Image to NumPy array in RGB format
        if isinstance(image, Image.Image):  # Check if the image is a PIL.Image
            image = np.array(image.convert("RGB"))
        
        face_encodings = face_recognition.face_encodings(image)
        if face_encodings:
            return face_encodings[0]  # Return the first face's embedding
        return None