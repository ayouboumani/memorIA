import os
import numpy as np
import shutil
import torch
from flask import Flask, request, render_template, send_from_directory
from transformers import AutoModel, AutoProcessor
from encode_face import process_images_in_folder, generate_embeddings_deepface
from encode_clip import encode_images 
from search import main

app = Flask(__name__)

# Define your load functions here (load_clip_model, load_clip_embeddings, etc.)...

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get uploaded photo and text query
        uploaded_file = request.files.get('photo')
        text_query = request.form.get('text_query')
        folder_path = request.form.get('folder_path')

        photo_folder = folder_path  # Dynamically get the folder path
        embedding_folder = os.path.join(folder_path, 'embeddings')  # Assume embeddings are in the same folder
        output_folder = './static/output'

        if os.path.exists(output_folder): 
            shutil.rmtree(output_folder)  # Clear previous output
        os.makedirs(output_folder, exist_ok=True)

        input_photo_path = None

        if uploaded_file and uploaded_file.filename != '':
            # Save the uploaded file in the output folder
            input_photo_path = os.path.join(output_folder, uploaded_file.filename)
            uploaded_file.save(input_photo_path)

        # Call the main processing function to find matches
        main(photo_folder, embedding_folder, output_folder, text_query,input_photo_path)

        # List the output photos
        output_photos = os.listdir(output_folder)

        # Remove the input photo from the matched photos if it exists
        if input_photo_path and uploaded_file.filename in output_photos:
            output_photos.remove(uploaded_file.filename)

        return render_template(
                'index.html',
                output_photos=output_photos,
                input_photo=uploaded_file.filename if input_photo_path else None,
                folder_path=folder_path,  # Keep the folder path
                text_query=text_query  # Keep the text query
                )

    return render_template('index.html', output_photos=None, input_photo=None)







@app.route('/output/<filename>')
def output_file(filename):
    return send_from_directory('static/output', filename)

if __name__ == "__main__":
    app.run(debug=True)
