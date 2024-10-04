import os
import time
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

# Load the Jina CLIP model and processor using AutoModel
model_id = "jinaai/jina-clip-v1"
model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# Function to load, preprocess, and rescale an image
def load_and_preprocess_image(image_path, target_size=None):
    image = Image.open(image_path).convert("RGB")  # Ensure it's RGB
    if target_size:
        height, width = image.size[:2]
        if width > target_size:
            scaling_factor = target_size / width
        image = image.resize((int(width * scaling_factor), int(height * scaling_factor)), Image.Resampling.LANCZOS)  # Rescale image to target size
    return image

# Function to search for images in a folder using a text query and measure average time
def search_images_in_folder(folder_path, text_query):
    image_scores = []
    times = []
    
    # Encode the text query
    inputs = processor(text=text_query, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    text_embeddings = model.get_text_features(**inputs)

    # Iterate through images in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.JPG', '.jpeg')):  # Add more extensions if needed
            image_path = os.path.join(folder_path, filename)
            image = load_and_preprocess_image(image_path)
            image_inputs = processor(images=image, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
            
            # Measure time for generating image embeddings
            start_time = time.time()
            image_embeddings = model.get_image_features(**image_inputs)
            end_time = time.time()
            
            # Append the time taken for this call
            times.append(end_time - start_time)

            # Calculate cosine similarity
            cosine_similarity = torch.nn.functional.cosine_similarity(text_embeddings, image_embeddings)
            image_scores.append((cosine_similarity.item(), filename))

    # Calculate the average time for generating image embeddings
    average_time = sum(times) / len(times) if times else 0

    # Sort images based on similarity scores in descending order
    image_scores.sort(reverse=True, key=lambda x: x[0])

    return image_scores, average_time

# Define the folder path and the text query
folder_path = "/home/ayoub/Pictures/mehdi"  # Replace with the actual path to your folder
text_query = "photo of me reading a book"  # Modify this query as needed

# Search for images and measure average time
results, avg_time = search_images_in_folder(folder_path, text_query)

# Display results
print("Search Results:")
for score, filename in results:
    print(f"{filename}: Score = {score:.4f}")

# Display average time taken to generate image embeddings
print(f"Average time for generating image embeddings: {avg_time:.4f} seconds")
