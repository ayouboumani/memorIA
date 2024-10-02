import os
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

# Load the Jina CLIP model and processor using AutoModel
model_id = "jinaai/jina-clip-v1"
model = AutoModel.from_pretrained(model_id,trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_id,trust_remote_code=True)

# Function to load and preprocess an image
def load_and_preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")  # Ensure it's RGB
    return image

# Function to search for images in a folder using a text query
def search_images_in_folder(folder_path, text_query):
    image_scores = []
    
    # Encode the text query
    inputs = processor(text=text_query, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    text_embeddings = model.get_text_features(**inputs)

    # Iterate through images in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.JPG', '.jpeg')):  # Add more extensions if needed
            image_path = os.path.join(folder_path, filename)
            image = load_and_preprocess_image(image_path)
            image_inputs = processor(images=image, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
            image_embeddings = model.get_image_features(**image_inputs)

            # Calculate cosine similarity
            cosine_similarity = torch.nn.functional.cosine_similarity(text_embeddings, image_embeddings)
            image_scores.append((cosine_similarity.item(), filename))

    # Sort images based on similarity scores in descending order
    image_scores.sort(reverse=True, key=lambda x: x[0])

    return image_scores

# Define the folder path and the text query
folder_path = "/home/ayoub/Pictures/test"  # Replace with the actual path to your folder
text_query = "photo of me reading a book"  # Modify this query as needed

# Search for images
results = search_images_in_folder(folder_path, text_query)

# Display results
print("Search Results:")
for score, filename in results:
    print(f"{filename}: Score = {score:.4f}")