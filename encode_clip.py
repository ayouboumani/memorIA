import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image


def encode_images(model, processor, input_folder, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Process each image in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith(('.jpg', '.jpeg', '.png', 'JPG')):  # Add other formats if needed
            embedding_file = os.path.join(output_folder, f"{file_name}_clip.npy")
            if os.path.exists(embedding_file):
                continue
            image_path = os.path.join(input_folder, file_name)

            # Load and process the image
            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt", padding=True)

            # Generate the image embedding
            with torch.no_grad():
                embeddings = model.get_image_features(**inputs)

            # Normalize the embeddings
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

            # Save the embeddings as a numpy file
            np.save(embedding_file, embeddings.cpu().numpy())

if __name__ == "__main__":
    input_folder = '/home/ayoub/Pictures/2024_Croitie'  # Update with your input folder path
    output_folder = '/home/ayoub/Pictures/2024_Croitie/embeddings'  # Update with your output folder path

    model_id = "jinaai/jina-clip-v1"
    model = AutoModel.from_pretrained(model_id,trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_id,trust_remote_code=True)
    encode_images(model, processor, input_folder, output_folder)
