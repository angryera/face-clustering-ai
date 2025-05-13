import sys
import time
from datetime import datetime
from data.dataset import Dataset
from models.model import Model
from utils.helpers import log, save_model
from face_recognition_handler import FaceRecognition
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from concurrent.futures import ProcessPoolExecutor

def log_with_timestamp(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log(f"[{timestamp}] {message}")

def visualize_clusters(embeddings, labels):
    log_with_timestamp("Visualizing clusters...")
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    unique_labels = set(labels)
    for label in unique_labels:
        label_mask = labels == label
        if label == -1:
            color = 'k'  # Black for outliers
            label_name = "Unknown"
        else:
            color = plt.cm.jet(float(label) / max(unique_labels))
            label_name = f"Person {label + 1}"
        plt.scatter(reduced_embeddings[label_mask, 0], reduced_embeddings[label_mask, 1], 
                    c=[color], label=label_name, alpha=0.6, edgecolors='w')
        # Add label names to the chart
        for x, y in reduced_embeddings[label_mask]:
            plt.text(x, y, label_name, fontsize=9, alpha=0.75)
    
    plt.title("Face Embedding Clusters")
    plt.legend()
    plt.show()

def extract_embedding(face_recognition, filename, image):
    log_with_timestamp(f"Extracting face embeddings for: {filename}")
    embedding = face_recognition.get_face_embedding(image)
    return (filename, image, embedding)

def extract_embedding_safe(args):
    try:
        # Reinitialize FaceRecognition for each process to ensure process safety
        face_recognition = FaceRecognition()
        filename, image = args
        log_with_timestamp(f"Extracting face embeddings for: {filename}")
        embedding = face_recognition.get_face_embedding(image)
        return (filename, image, embedding)
    except Exception as e:
        log_with_timestamp(f"Error extracting embedding for {filename}: {e}")
        return (filename, image, None)

def main():
    start_time = time.time()
    log_with_timestamp("Initializing the application...")
    
    # Initialize face recognition
    face_recognition = FaceRecognition()
    
    # Create a Dataset instance
    dataset = Dataset(file_path=None)  # No file path needed for this use case
    
    # Load images to categorize
    images = dataset.load_images_from_folder("imgs/original")  # Update this path
    
    # Extract face embeddings for clustering using parallelization
    log_with_timestamp("Extracting face embeddings in parallel using ProcessPoolExecutor...")
    try:
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(
                extract_embedding_safe, [(img[0], img[1]) for img in images]
            ))
    except Exception as e:
        log_with_timestamp(f"Error during parallel embedding extraction: {e}")
        sys.exit(1)
    
    embeddings = []
    filenames = []
    for filename, image, embedding in results:
        if embedding is not None:
            embeddings.append(embedding)
            filenames.append((filename, image))
        else:
            log_with_timestamp(f"Embedding for {filename} is None.")
    
    if not embeddings:
        log_with_timestamp("No embeddings were extracted. Exiting.")
        sys.exit(1)
    
    # Perform clustering with refined values
    log_with_timestamp("Clustering face embeddings with refined parameters around eps=0.5 and min_samples=2...")
    embeddings = np.array(embeddings)
    clustering_params = [
        {"eps": 0.5, "min_samples": 1},
    ]
    
    for i, params in enumerate(clustering_params):
        log_with_timestamp(f"Using DBSCAN with eps={params['eps']} and min_samples={params['min_samples']}...")
        clustering = DBSCAN(metric="euclidean", **params).fit(embeddings)
        labels = clustering.labels_
        
        # Visualize clusters
        visualize_clusters(embeddings, labels)
        
        # Categorize images based on clusters
        base_dir = f"imgs/categorized/refined_experiment_{i+1}"  # Directory for this parameter set
        for (filename, image), label in zip(filenames, labels):
            if label == -1:
                save_path = f"{base_dir}/unknown/{filename}"  # Save outliers
            else:
                save_path = f"{base_dir}/Person{label + 1}/{filename}"  # Save clusters
            
            Dataset.save_image(image, save_path)
            log_with_timestamp(f"Saved {filename} to {save_path}")
    
    log_with_timestamp("Categorization completed.")
    end_time = time.time()
    total_time = end_time - start_time
    log_with_timestamp(f"Total execution time: {total_time:.2f} seconds.")

if __name__ == "__main__":
    main()