"""
Flask app for processing images.

This script provides two endpoints:
1. /identify: Processes an image and returns the top-k identities.
2. /add: Adds a provided image to the gallery with an associated name.

Usage:
    Run the app with: python app.py
    Sample curl command for /identify:
        curl -X POST -F "image=@/path/to/image.jpg" -F "k=3" http://localhost:5000/identify
    Sample curl command for /add:
        curl -X POST -F "image=@/path/to/image.jpg" -F "name=Firstname_Lastname" http://localhost:5000/add
"""
import os.path

import numpy as np
import torch
from PIL import Image
from flask import Flask, request, jsonify

from modules.extraction.embedding import Embedding
from modules.extraction.preprocessing import Preprocessing
from modules.retrieval.index.hnsw import FaissHNSW
from modules.retrieval.search import FaissSearch

app = Flask(__name__)

## List of designed parameters: 
# (Configure these parameters according to your design decisions)
DEFAULT_K = '3'
MODEL = 'vggface2'
INDEX = 'hnsw'
SIMILARITY_MEASURE = 'cosine'
SAVE_INDEX = False
INDEX_PATH = 'scripts/faiss_hnsw_index.pkl'
# Add more if needed...
preprocessor = Preprocessing()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Embedding(pretrained=MODEL, device=device)
index = None
if SAVE_INDEX and os.path.exists(INDEX_PATH):
    index = FaissHNSW.load(INDEX_PATH) # Load data from index
else:
    index = FaissHNSW(dim=512, metric=SIMILARITY_MEASURE)
search = FaissSearch(index, metric=SIMILARITY_MEASURE)

@app.route('/identify', methods=['POST'])
def identify():
    """
    Process the probe image to identify top-k identities in the gallery.

    Expects form-data with:
      - image: Image file to be processed.
      - k: (optional) Integer specifying the number of top identities 
           (default is 3).

    Returns:
      JSON response with a success message and the provided value of k.
      If errors occur, returns a JSON error message with the appropriate status code.
    """
    # Check if the request has the image file
    if 'image' not in request.files:
        return jsonify({"error": "No image part in the request"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    # Retrieve and validate the integer parameter "k"
    try:
        k = int(request.form.get('k', DEFAULT_K))
    except ValueError:
        return jsonify({"error": "Invalid integer for parameter 'k'"}), 400

    # Convert the image into a NumPy array
    try:
        image = np.array(Image.open(file))
        print(image)
    except Exception as e:
        return jsonify({
            "error": "Failed to convert image to numpy array",
            "details": str(e)
        }), 500

    ########################################
    # TASK 1a: Implement /identify endpoint
    #         to return the top-k identities
    #         of the provided probe.
    ########################################
    image_tensor = preprocessor.process(Image.fromarray(image))
    embedding = model.encode(image_tensor)
    distances, indices, metadatas = search.search(embedding, k)
    print("distances: ", distances)
    print("indices: ", indices)
    print("metadatas: ", metadatas)
    identities = [name for name in metadatas[0] if name is not None]
    return jsonify({
        "message": f"Returned top-{k} identities",
        "ranked identities": identities
    }), 200


@app.route("/add", methods=['POST'])
def add():
    """
    Add a provided image to the gallery with an associated name.

    Expects form-data with:
      - image: Image file to be added.
      - name: String representing the identity associated with the image.

    Returns:
      JSON response confirming the image addition.
      If errors occur, returns a JSON error message with the appropriate status code.
    """
    # Check if the request has the image file
    if 'image' not in request.files:
        return jsonify({"Error": "No image part in the request"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"Error": "No file selected for uploading"}), 400

    # Convert the image into a NumPy array
    try:
        image = np.array(Image.open(file))
        print(image)
    except Exception as e:
        return jsonify({
            "error": "Failed to convert image to numpy array",
            "details": str(e)
        }), 500

    # Retrieve the 'name' parameter
    name = request.form.get('name')
    if not name:
        return jsonify({"Error": "Must have associated 'name'"}), 400

    ########################################
    # TASK 1b: Implement `/add` endpoint to
    #         add the provided image to the 
    #         catalog.
    ########################################
    if name in index.metadata:
        return jsonify({"Error": f"The image {name} already existed."}), 400

    image_tensor = preprocessor.process(Image.fromarray(image))
    embedding = model.encode(image_tensor)
    index.add_embeddings([embedding], [name])

    if SAVE_INDEX:
        index.save(INDEX_PATH)

    return jsonify({
        "message": f"New image added to gallery (as {name}) and indexed into catalog."
    })


if __name__ == '__main__':
    app.run(port=5000, debug=True, host='0.0.0.0')
