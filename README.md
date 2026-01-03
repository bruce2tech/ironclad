# Ironclad - Face Recognition and Retrieval System

A high-performance face recognition system implementing multiple similarity search algorithms with comprehensive benchmarking capabilities. This project was developed as part of a graduate course in AI-enabled systems at Johns Hopkins University.

## Overview

Ironclad is a Flask-based face recognition and retrieval system that leverages deep learning embeddings and advanced indexing methods to efficiently identify faces from a gallery. The system supports multiple embedding models, similarity metrics, and indexing strategies including brute-force search, HNSW (Hierarchical Navigable Small World), and LSH (Locality-Sensitive Hashing).

## Key Features

- **Multiple Face Embedding Models**: Support for VGGFace2, FaceNet, and other state-of-the-art face recognition models
- **Advanced Indexing Methods**:
  - Brute-force search for baseline comparisons
  - FAISS HNSW for fast approximate nearest neighbor search
  - LSH for scalable similarity search
- **Flexible Similarity Metrics**: Cosine similarity and Euclidean distance
- **REST API**: Easy-to-use Flask endpoints for face identification and gallery management
- **Comprehensive Benchmarking Suite**: Extensive tools for evaluating performance under various conditions

## System Architecture

The system is organized into modular components:

```
ironclad/
├── modules/
│   ├── extraction/       # Face embedding extraction
│   ├── retrieval/        # Search and indexing algorithms
│   └── preprocessing/    # Image preprocessing pipeline
└── app.py               # Flask API server
```

## API Endpoints

### POST /identify
Identifies a person from an uploaded image.

**Parameters:**
- `image`: Image file to identify
- `k`: Number of top matches to return (default: 3)

**Example:**
```bash
curl -X POST -F "image=@photo.jpg" -F "k=3" http://localhost:5000/identify
```

### POST /add
Adds a new face to the gallery.

**Parameters:**
- `image`: Image file to add
- `name`: Name associated with the face

**Example:**
```bash
curl -X POST -F "image=@person.jpg" -F "name=John_Doe" http://localhost:5000/add
```

## Demo Data

The repository includes a small sample dataset in `demo_data/` for quick testing and demonstrations:

```
demo_data/
├── gallery/          # Gallery images (5 people)
│   ├── Aaron_Sorkin/
│   ├── Abdullah_Gul/
│   ├── Adam_Scott/
│   ├── Abel_Pacheco/
│   └── Adolfo_Rodriguez_Saa/
└── queries/          # Query images for testing
    ├── Aaron_Sorkin/
    ├── Abdullah_Gul/
    └── Adam_Scott/
```

**For larger datasets:** The full benchmarking suite was tested on CASIA-WebFace and VGGFace2 datasets. You can download these public datasets separately and place them in the `chips/` directory.

## Benchmarking and Evaluation

This project includes comprehensive benchmarking tools for comparing different retrieval methods:

### Performance Comparison Scripts
- `compare_bruteforce_vs_hnsw.py` - Compare brute-force vs HNSW performance
- `compare_hnsw_vs_lsh_vggface2.py` - Compare HNSW vs LSH on VGGFace2
- `compare_bruteforce_vs_hnsw_vs_lsh_vggface2.py` - Three-way comparison

### Robustness Evaluation
- `evaluate_gaussian_noise_retrieval.py` - Test performance with noisy images
- `evaluate_resize_retrieval.py` - Test performance with different image resolutions
- `evaluate_retrieval.py` - General retrieval performance metrics

### Model Comparison
- `compare_models.py` - Compare different embedding models
- `compare_distances_by_model.py` - Analyze distance metrics across models

### Parameter Tuning
- `sweep_m_casia.py` - HNSW parameter tuning on CASIA dataset
- `sweep_m_vggface2.py` - HNSW parameter tuning on VGGFace2 dataset

## Utility Scripts

- `embed_dir_to_npz.py` - Pre-compute embeddings for a directory of images
- `mirror_resize_images.py` - Resize images while maintaining aspect ratio
- `mirror_brightness_images.py` - Adjust image brightness for testing

## Technologies Used

- **Deep Learning**: PyTorch, FaceNet, VGGFace2
- **Similarity Search**: FAISS (HNSW), LSH
- **Web Framework**: Flask
- **Image Processing**: PIL, OpenCV
- **Scientific Computing**: NumPy, SciPy

## Quick Start

### Option 1: Interactive Demo (Recommended for First-Time Users)

Try the Jupyter notebook demo with sample data included in the repository:

```bash
# Clone the repository
git clone https://github.com/bruce2tech/ironclad.git
cd ironclad

# Install dependencies
pip install torch torchvision pillow numpy faiss-cpu flask matplotlib scikit-learn jupyter

# Launch the demo notebook
jupyter notebook demo.ipynb
```

The demo notebook includes:
- Sample face images (5 people, 19 images total)
- Step-by-step walkthrough of face recognition
- Performance comparisons between search methods
- Embedding space visualization

### Option 2: Full Installation

For production use with your own datasets:

```bash
# Clone the repository
git clone https://github.com/bruce2tech/ironclad.git
cd ironclad

# Install dependencies
pip install -r requirements.txt

# Run the Flask server
python -m ironclad.app
```

## Performance Highlights

The system has been extensively benchmarked on:
- **CASIA-WebFace** dataset
- **VGGFace2** dataset

Key findings from benchmarking:
- HNSW provides significant speedup over brute-force with minimal accuracy loss
- System maintains robust performance under various image quality conditions
- Optimal parameters vary by dataset and use case

## Project Context

This project was developed as part of a graduate course in Creating AI-Enabled Systems at Johns Hopkins University. It demonstrates practical implementation of face recognition systems, similarity search algorithms, and comprehensive performance evaluation methodologies.

## Attribution

This repository originated from a course project at Johns Hopkins University. While the course provided initial starter code and project specifications, the majority of the implementation represents significant original work beyond the base requirements.

### Original Contributions (Patrick Bruce):

**Benchmarking & Performance Analysis:**
- Complete benchmarking suite (`compare_*.py`, `evaluate_*.py`, `benchmark_*.py`)
- Performance comparison framework (HNSW vs LSH vs Brute-Force)
- Robustness testing (noise, resize, brightness variations)
- Parameter tuning and optimization scripts
- Statistical analysis and visualization tools

**Advanced Features:**
- LSH (Locality-Sensitive Hashing) implementation and integration
- Enhanced HNSW parameter optimization
- Multi-model comparison framework
- Embedding pre-computation utilities
- Extended preprocessing pipeline

**Documentation & Demo:**
- Interactive Jupyter notebook demo
- Comprehensive README documentation
- Sample dataset curation
- API usage examples

**Code Enhancements:**
- Enhanced embedding extraction modules
- Improved search algorithms
- Additional utility scripts
- Performance optimization

### Course-Provided Base Components:
- Initial project structure and specifications
- Base Flask API framework
- Core module interfaces
- Assignment requirements

**Note:** The extensive benchmarking suite, performance analysis tools, and demo materials demonstrate work that significantly extends beyond the original course requirements.

## Author

Patrick Bruce

## License

This project is for educational and portfolio purposes.
