import numpy as np
import os
import cv2
import shutil

def load_face_embeddings(folder_path):
    embeddings = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.npy'):
            # Load the embedding
            embedding = np.load(os.path.join(folder_path, file_name))
            # Use the filename (without extension) as the key
            embeddings[file_name[:-4]] = embedding
    return embeddings

def cosine_similarity(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2)
    norm_a = np.linalg.norm(embedding1)
    norm_b = np.linalg.norm(embedding2)
    if norm_a == 0 or norm_b == 0:
        return 0.0  # Avoid division by zero
    return dot_product / (norm_a * norm_b)

def find_closest_matches(ref_embedding, embeddings, threshold=0.9):
    closest_matches = []
    for file_name, embedding in embeddings.items():
        similarity = cosine_similarity(ref_embedding, embedding)
        if similarity > threshold:
            closest_matches.append((file_name, similarity))
    # Sort by similarity
    closest_matches.sort(key=lambda x: x[1], reverse=True)  # Higher similarity comes first
    return closest_matches

def save_matched_photos(closest_matches, output_folder, original_folder):
    os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist
    for file_name, _ in closest_matches:
        file_name = file_name.replace(".JPG","")
        original_file_path = os.path.join(original_folder, f"{file_name}.jpg")  # Assuming original photos are JPEGs
        if os.path.exists(original_file_path):
            shutil.copy(original_file_path, output_folder)  # Copy the photo to the output folder
        else:
            print(f"Original photo {original_file_path} does not exist.")

def main(ref_embedding_path, embeddings_folder, output_folder, original_folder, threshold=0.9):
    # Load the reference embedding
    ref_embedding = np.load(ref_embedding_path)

    # Load all face embeddings
    embeddings = load_face_embeddings(embeddings_folder)

    # Find closest matches
    closest_matches = find_closest_matches(ref_embedding, embeddings, threshold)

    # Save matched photos
    save_matched_photos(closest_matches, output_folder, original_folder)

    # Output results
    if closest_matches:
        print("Closest matches:")
        for file_name, distance in closest_matches:
            print(f"Photo: {file_name}, Distance: {distance:.4f}")
    else:
        print("No matches found within the threshold.")

if __name__ == "__main__":
    # Paths to the reference embedding and folder containing face embeddings
    ref_embedding_path = '/home/ayoub/Pictures/2024_Croitie/embeddings/P1123508.JPG_face0.npy'  # Update with your reference embedding path
    folder_path = '/home/ayoub/Pictures/2024_Croitie/embeddings'  # Update with your embeddings folder path
    output_folder = '/home/ayoub/Pictures/2024_Croitie/output'  # Update with your output folder path
    original_folder = '/home/ayoub/Pictures/2024_Croitie/faces'  # Update with the folder containing the original photos

    # Threshold for considering a match
    threshold = 0.5  # You can adjust this value

    if os.path.exists(output_folder): 
        shutil.rmtree(output_folder)
    main(ref_embedding_path, folder_path,output_folder, original_folder, threshold)
