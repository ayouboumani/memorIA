import os
import numpy as np
import shutil
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


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
            clip_embeddings[key] = embedding
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
                face_embeddings[key] = embedding
    return face_embeddings

def encode_text(query, model, processor):
    # Encode the text query
    inputs = processor(text=query, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_embedding = model.get_text_features(**inputs)
    return text_embedding / text_embedding.norm(dim=-1, keepdim=True)  # Normalize

def find_similar_photos(face_embedding, faces_embeddings, photo_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    matched_photos = []

    # Compute similarity for each CLIP embedding
    for key in faces_embeddings:
        # Calculate cosine similarity between CLIP embedding and text embedding
        similarity = torch.cosine_similarity(torch.tensor(face_embedding).unsqueeze(0), torch.tensor(faces_embeddings[key]).unsqueeze(0)).item()
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

    # Load CLIP embeddings
    clip_embeddings = load_clip_embeddings(embedding_folder)

    # Load Face embeddings
    face_embeddings = load_face_embeddings(embedding_folder)

    # Encode the text query
    text_embedding = encode_text(text_query, model, processor)

    # Filter photos based on text query embeddings
    filtered_photos = {key: embedding for key, embedding in clip_embeddings.items() if torch.cosine_similarity(torch.tensor(embedding), text_embedding).item() > 0.2}  # Adjust threshold as needed

    if not filtered_photos:
        print("No photos matched the text query.")
        return

    print(f"Filtered photos based on text query: {filtered_photos.keys()}")

    # Load the input photo face embedding
    input_face_embedding = np.load(input_photo)  # Assuming input_photo is the path to the face embedding

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
    photo_folder = '/home/ayoub/Pictures/2024_Croitie'  # Update with your folder containing original photos
    embedding_folder = '/home/ayoub/Pictures/2024_Croitie/embeddings'  # Update with your folder containing embeddings
    output_folder = '/home/ayoub/Pictures/2024_Croitie/output'  # Update with your output folder path
    input_photo = '/home/ayoub/Pictures/2024_Croitie/embeddings/P1123508.JPG_face0.npy'  # Update with the input photo path
    text_query = "photo of me with sunglasses"  # Example text query

    if os.path.exists(output_folder): 
        shutil.rmtree(output_folder)


    # Find the closest photo
    main(photo_folder, embedding_folder, output_folder, input_photo, text_query)
