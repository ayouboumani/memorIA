import os
from deepface import DeepFace
from PIL import Image

# Function to resize images for efficient processing
def resize_image(image_path, max_size=(400, 300)):
    image = Image.open(image_path)
    image.thumbnail(max_size)
    resized_path = "resized_" + os.path.basename(image_path)
    image.save(resized_path)
    return resized_path

# Function to verify if the user's face is in the folder's images
def verify_user_in_folder(user_image_path, folder_path, model_name="VGG-Face"):
    # Load the model once
    print("Loading model...")
    DeepFace.build_model(model_name)  # This preloads the model

    matches = []
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        # Check if the file is an image
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        # Resize images to reduce memory usage
        resized_user_image = resize_image(user_image_path)
        resized_image = resize_image(image_path)
        
        try:
            # Perform face verification
            result = DeepFace.verify(img1_path=resized_user_image, img2_path=resized_image, model_name=model_name, enforce_detection=False)
            if result['verified']:
                print(f"Match found: {image_name}")
                matches.append(image_name)
            else:
                print(f"No match: {image_name}")
        except Exception as e:
            print(f"Error processing {image_name}: {e}")
    
    return matches

# Usage
folder_path = "/home/ayoub/Pictures/2024_Croitie"  # Replace with your folder path
user_image_path = "/home/ayoub/Pictures/2024_Croitie/P1123685.JPG" 


matches = verify_user_in_folder(user_image_path, folder_path)
if matches:
    print(f"Matches found: {matches}")
else:
    print("No matches found.")


