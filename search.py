import os
import numpy as np
import shutil
import torch
import concurrent.futures
import time
from PIL import Image
from transformers import AutoModel, AutoProcessor
from encode_face import process_images_in_folder, generate_embeddings_deepface
from encode_clip import encode_images 

def load_clip_model():
    # Load the Jina CLIP model 
    model_id = "jinaai/jina-clip-v1"
    model = AutoModel.from_pretrained(model_id,trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_id,trust_remote_code=True)
    return model, processor


def load_clip_embeddings(folder):
    clip_embeddings = {}
    for file_name in os.listdir(folder):
        if file_name.endswith('_clip.npy'):
            # Load CLIP embeddings
            clip_path = os.path.join(folder, file_name)
            embedding = np.load(clip_path)
            key = os.path.basename(file_name).replace('_clip.npy', '')
            clip_embeddings[key] = torch.Tensor(embedding)
    return clip_embeddings

def load_face_embeddings(folder):
    face_embeddings = {}
    for file_name in os.listdir(folder):
        if file_name.endswith('_face0.npy'):
            for i in range(100):
                face_path = os.path.join(folder, file_name.replace('_face0.npy',f'_face{i}.npy'))
                if not os.path.exists(face_path):
                    break
                # Load face embeddings
                embedding = np.load(face_path)
                key = os.path.basename(face_path)[:-4]
                face_embeddings[key] = torch.Tensor(embedding)
    return face_embeddings

def encode_text(query, model, processor):
    # Encode the text query
    inputs = processor(text=query, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_embedding = model.get_text_features(**inputs)
    return text_embedding / text_embedding.norm(dim=-1, keepdim=True)  # Normalize

def get_top_k_similar_photos(clip_embeddings, text_embedding, k=5):
    # Compute the cosine similarity between each image embedding and the text embedding
    similarities = {key: torch.cosine_similarity(embedding, text_embedding).item()
                    for key, embedding in clip_embeddings.items()}

    # Sort by similarity in descending order and select the top k
    top_k = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:k]

    # Filter the embeddings of the top k similar photos
    filtered_photos = {key: clip_embeddings[key] for key, _ in top_k}
    
    return filtered_photos

def find_similar_photos(face_embedding, faces_embeddings, photo_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    matched_photos = []

    # Compute similarity for each CLIP embedding
    for key in faces_embeddings:
        # Calculate cosine similarity between CLIP embedding and text embedding
        similarity = torch.cosine_similarity(face_embedding.unsqueeze(0), faces_embeddings[key].unsqueeze(0)).item()
        print(f"Similarity for {key}: {similarity}")

        if similarity > 0.1:  # Threshold for similarity
            matched_photos.append(key)

    # Copy matched photos to the output folder
    for photo in matched_photos:
        output_photo_path = os.path.join(photo_folder, photo.split('_')[0])  # Adjust the extension if needed
        if os.path.exists(output_photo_path):
            output_path = os.path.join(output_folder, os.path.basename(output_photo_path))
            shutil.copy(output_photo_path, output_path)  # Copy the photo
            print(f"Copied {output_photo_path} to {output_path}")
        else:
            print(f"Original photo {output_photo_path} does not exist.")

    if not matched_photos:
        print("No similar photos found.")

def main(photo_folder, embedding_folder, output_folder, input_photo, text_query):
    # Load CLIP model
    model, processor = load_clip_model()

    start_time = time.time()  # Record the start time

    process_images_in_folder(photo_folder, embedding_folder)
    encode_images(photo_folder, embedding_folder)

    end_time = time.time()  # Record the end time
    total_time = end_time - start_time  # Calculate the total time taken
    print(f"Total time for computing embeddings: {total_time:.2f} seconds")


    
    # Load CLIP embeddings
    clip_embeddings = load_clip_embeddings(embedding_folder)

    # Load Face embeddings
    face_embeddings = load_face_embeddings(embedding_folder)

    # Encode the text query
    text_embedding = encode_text(text_query, model, processor)

    # Encode query face
    input_face_embeddings, _  = generate_embeddings_deepface(input_photo)
    input_face_embedding = torch.Tensor(input_face_embeddings[0])

    # Filter photos based on text query embeddings
    filtered_photos = get_top_k_similar_photos(clip_embeddings, text_embedding)

    if not filtered_photos:
        print("No photos matched the text query.")
        return

    print(f"Filtered photos based on text query: {filtered_photos.keys()}")


    filtered_faces = {}
    for key in filtered_photos:
        face_keys = [key+f"_face{i}" for i in range(100)]
        for face_key in face_keys:
            if not face_key in face_embeddings:
                break
            filtered_faces[face_key] = face_embeddings[face_key]


    # Find similar photos from the filtered results
    find_similar_photos(input_face_embedding, filtered_faces, photo_folder, output_folder)



if __name__ == "__main__":
    photo_folder = '/home/ayoub/Pictures/2024_ValdAzun'  # Update with your folder containing original photos
    embedding_folder = '/home/ayoub/Pictures/2024_ValdAzun/embeddings'  # Update with your folder containing embeddings
    output_folder = '/home/ayoub/Pictures/2024_ValdAzun/output'  # Update with your output folder path
    input_photo = '/home/ayoub/Pictures/2024_ValdAzun/P1112792.JPG'  # Update with the input photo path
    text_query = "a photo of me dancing"  # Example text query

    if os.path.exists(output_folder): 
        shutil.rmtree(output_folder)
    
    os.makedirs(embedding_folder, exist_ok=True)


    # Find the closest photo
    main(photo_folder, embedding_folder, output_folder, input_photo, text_query)
