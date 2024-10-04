import os
import cv2
import numpy as np
from deepface import DeepFace
from deepface.modules import detection
from sklearn.metrics.pairwise import cosine_similarity

# 1. Resize image to a manageable size to avoid memory issues
def resize_image(image, target_width=800):
    height, width = image.shape[:2]
    if width > target_width:
        scaling_factor = target_width / width
        resized_image = cv2.resize(image, (int(width * scaling_factor), int(height * scaling_factor)))
        return resized_image
    return image

# 2. Load image paths from a parent folder
def get_image_paths(folder):
    image_paths = []
    for file in os.listdir(folder):
        if file.endswith(('.jpg', '.jpeg', '.png', 'JPG')):  # Add other formats if needed
            image_paths.append(os.path.join(folder, file))
    return image_paths

# 3. Generate face embeddings using DeepFace for each face detected in the image
def generate_embeddings_deepface(image_path, align=False):
    try:
        # Read and resize the image to avoid memory issues
        img = cv2.imread(image_path)
        img_resized = resize_image(img)

        # Detect faces in the resized image using the detector DeepFace uses internally
        detected_faces = detection.detect_faces('mtcnn', img_resized)

        embeddings, faces = [], []
        if len(detected_faces) > 0:
            for face in detected_faces:
                face_img = face.img
                embedding = DeepFace.represent(img_path=face_img, model_name="Facenet", enforce_detection=False, align=align)
                embeddings.append(embedding[0]["embedding"])
                faces.append(face_img)
        return embeddings, faces  # Return a list of embeddings for each detected face
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# 4. Save each face's embedding to a .npy file
def save_embedding(embedding, output_path, face_index):
    np.save(f"{output_path}_face{face_index}.npy", embedding)

def save_image(image_array, filename):
    """
    Save an image array as an image file.

    Parameters:
    - image_array: numpy array, the image data to be saved.
    - filename: str, the path where the image will be saved (with extension).
    """
    # Ensure the image array is in the correct format
    if isinstance(image_array, np.ndarray):
        # Save the image
        cv2.imwrite(filename, image_array)
        print(f"Image saved as {filename}")
    else:
        print("Invalid image array format. Please provide a numpy array.")


# 5. Check if embedding already exists
def embedding_exists(output_path, face_index):
    return os.path.exists(f"{output_path}_face{face_index}.npy")

# 6. Compare two embeddings using cosine similarity
def compare_embeddings_cosine(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0][0]

# 7. Compare two embeddings using Euclidean distance
def compare_embeddings_euclidean(embedding1, embedding2):
    return np.linalg.norm(np.array(embedding1) - np.array(embedding2))

# 8. Full workflow: generate embeddings for all detected faces in each image and save them for later use
def process_images_in_folder(parent_folder, output_folder, align_faces=True, selection = None):
    if selection is None:
        image_paths = get_image_paths(parent_folder)
    else:
        image_paths = [os.path.join(parent_folder,image_name) for image_name in selection]
    # Load the already processed images from the log file
    log_file = os.path.join(output_folder,"log_faces.txt")
    if os.path.exists(log_file):
        with open(log_file, 'r') as log:
            processed_images = {line.strip() for line in log}  # Use a set for faster lookups
    else:
        processed_images = set()

    for img_path in image_paths:
        output_file = os.path.join(output_folder, os.path.basename(img_path))

        # Skip processing if the image is already logged as processed
        if img_path in processed_images:
            print(f"Skipping already processed image: {img_path}")
            continue

        embeddings, faces = generate_embeddings_deepface(img_path, align=align_faces)

        if embeddings is not None and len(embeddings) > 0:
            for i, embedding in enumerate(embeddings):
                save_embedding(embedding, output_file, i)
                print(f"Saved embedding for {img_path}, face {i}")
                save_image(faces[i], output_file[:-4].replace("embeddings","faces")+f"_face{i}.jpg")
        else:
            print(f"No faces detected in {img_path}")
        # Log the processed image name
        with open(log_file, 'a') as log:
            log.write(f"{img_path}\n")  # Append the processed image path to the log file


# Example usage
if __name__ == "__main__":
    # Folder containing your images
    parent_folder = "/home/ayoub/Pictures/2024_Croitie"
    
    # Folder where you want to save the embeddings
    output_folder = "/home/ayoub/Pictures/2024_Croitie/embeddings"
    os.makedirs(output_folder, exist_ok=True)
    
    # Process all images, generate embeddings for all faces, and save them
    process_images_in_folder(parent_folder, output_folder, align_faces=False)

