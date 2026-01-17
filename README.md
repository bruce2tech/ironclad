# IronClad: Production Face Recognition with Scalable Similarity Search

> High-accuracy face identification system achieving 96.5% Top-1 accuracy with sub-millisecond query times, demonstrating the engineering trade-offs between exact and approximate nearest neighbor search.

## The Problem

Face recognition systems in production environments face a fundamental tension: **accuracy vs. speed at scale**. A gallery of 1,000 faces is trivial—brute-force search works fine. But scale to 100,000+ identities (enterprise access control, law enforcement databases) and naive approaches collapse:

- **Brute-force**: O(n) search time, infeasible at scale
- **Approximate methods**: Fast, but how much accuracy do you sacrifice?
- **Embedding quality**: Model choice affects both accuracy AND robustness to real-world degradation (lighting, noise, resolution)

This project systematically answers these questions through rigorous benchmarking across embedding models, indexing strategies, and degraded conditions.

## Key Findings

### Model Selection: ArcFace Dominates Under Stress

I evaluated VGGFace2 and ArcFace embeddings across multiple degradation scenarios. The results informed a clear recommendation:

| Condition | ArcFace Top-1 | VGGFace2 Top-1 | Gap |
|-----------|-------------:|---------------:|----:|
| Baseline (clean) | 96.5% | 91.3% | +5.2% |
| High Noise (σ=0.1) | 93.9% | 82.4% | **+11.5%** |
| Low Resolution (0.25x) | 96.4% | 78.8% | **+17.6%** |
| Dark Lighting (0.5x) | 97.2% | 92.1% | +5.1% |

**Strategic Decision**: ArcFace's angular margin loss creates more robust embeddings. The 17.6% accuracy advantage under low resolution is critical for real-world deployments where image quality varies (surveillance cameras, mobile uploads).

### Indexing Strategy: When to Sacrifice Exactness

| Method | Query Time | Accuracy@1 | Recommended Scale |
|--------|----------:|----------:|----------|
| Brute-Force | ~0.01ms | 100% | <1K identities |
| HNSW (M=16) | ~0.07ms | 99.9% | 1K-100K identities |
| LSH | ~0.05ms | ~98% | >100K identities |

**Trade-off Analysis**: HNSW provides the optimal balance for most production systems. The 0.1% accuracy drop is negligible, while the graph-based structure scales logarithmically. LSH only becomes advantageous at extreme scale where memory constraints dominate.

### Parameter Tuning: HNSW M-Value Optimization

Systematic sweeps on CASIA-WebFace and VGGFace2 datasets revealed:

- **M=8**: Insufficient graph connectivity, accuracy degrades on diverse faces
- **M=16**: Sweet spot—99.9% accuracy with 7x speedup over brute-force
- **M=32**: Diminishing returns, memory overhead not justified

## System Architecture

```
ironclad/
├── app.py                      # Flask API server
└── modules/
    ├── extraction/             # Face embedding pipeline
    │   ├── embedding.py        # ArcFace/VGGFace2 models
    │   ├── preprocessing.py    # Face detection, alignment, normalization
    │   └── embedders.py        # Model architecture wrappers
    └── retrieval/              # Scalable search infrastructure
        ├── search.py           # Unified search interface
        └── index/
            ├── hnsw.py         # FAISS HNSW (production default)
            ├── bruteforce.py   # Exact search baseline
            └── lsh.py          # Extreme-scale option
```

## API Reference

### POST /identify
Identifies a person from an uploaded image against the gallery.

```bash
curl -X POST -F "image=@query.jpg" -F "k=3" http://localhost:5000/identify
```

**Response**:
```json
{
  "matches": [
    {"name": "John_Doe", "confidence": 0.94, "distance": 0.12},
    {"name": "Jane_Smith", "confidence": 0.67, "distance": 0.38}
  ],
  "query_time_ms": 0.07
}
```

### POST /add
Enrolls a new identity into the gallery.

```bash
curl -X POST -F "image=@person.jpg" -F "name=John_Doe" http://localhost:5000/add
```

## Quick Start

### Option 1: Interactive Demo (Recommended)

The demo notebook walks through face recognition concepts with included sample data:

```bash
git clone https://github.com/bruce2tech/ironclad.git
cd ironclad

pip install torch torchvision pillow numpy faiss-cpu flask matplotlib scikit-learn jupyter

jupyter notebook demo.ipynb
```

The demo includes:
- 5 identities with 16 gallery images
- Step-by-step retrieval walkthrough
- Embedding space visualization
- Search method comparison

### Option 2: Production Deployment

```bash
git clone https://github.com/bruce2tech/ironclad.git
cd ironclad

pip install -r requirements.txt

python -m ironclad.app
```
Open a second terminal to interact with the app. 

Note: The Flask app is an API-only backend, not a website with a web interface. There's no HTML page to view—it only responds to API requests, like the curl commands.

Terminal 1: Keep the server running with python -m ironclad.app

Terminal 2: Run the curl commands to hit the API

- Add the images in the "ironclad/"demo_data/gallery/" folder
- Use curl -X POST -F "image=@filepath/query.jpg" -F "k=3" http://localhost:5000/identify
- Query the images in the "ironclad/"demo_data/query/" folder
- Use curl -X POST -F "image=@filepath/person.jpg" -F "name=John_Doe" http://localhost:5000/add

## Evaluation Suite

The repository includes comprehensive benchmarking tools developed to answer specific engineering questions:

| Script | Purpose |
|--------|---------|
| `compare_bruteforce_vs_hnsw.py` | Quantify accuracy/speed trade-off |
| `compare_hnsw_vs_lsh_vggface2.py` | Evaluate approximate methods head-to-head |
| `evaluate_gaussian_noise_retrieval.py` | Stress-test robustness to image noise |
| `evaluate_resize_retrieval.py` | Measure resolution degradation impact |
| `sweep_m_*.py` | HNSW parameter optimization |

## Technical Insights

### Key Observations

1. **Embedding choice matters more than indexing**: The ArcFace vs VGGFace2 gap (17.6% under degradation) dwarfs the HNSW vs brute-force gap (0.1%). Investment in embedding quality yields greater returns than indexing optimization.

2. **"Approximate" is a misnomer**: HNSW's 99.9% accuracy demonstrates that approximate nearest neighbor search introduces negligible error for most applications. The engineering benefit of logarithmic scaling justifies this approach.

3. **Baseline accuracy is misleading**: Performance under degraded conditions (noise, poor lighting, low resolution) determines production viability. Clean-image benchmarks overstate real-world performance.

### Production Considerations

For deployment beyond this prototype:

- **Face detection pipeline**: Current implementation assumes cropped faces. Production requires MTCNN or RetinaFace preprocessing.
- **Gallery updates**: Hot-swapping embeddings without service restart requires index rebuild strategies.
- **Threshold calibration**: The 0.5 similarity threshold should be tuned per-deployment based on FAR/FRR requirements.
- **Hardware acceleration**: GPU inference for embedding extraction, potentially GPU-accelerated FAISS for extreme scale.

### Known Limitations

- Evaluated on controlled datasets (CASIA-WebFace, VGGFace2); real-world demographic diversity may affect performance
- Single-face assumption; multi-face detection pipeline not implemented
- No liveness detection (vulnerable to photo attacks)

## Technologies

- **Deep Learning**: PyTorch, ArcFace, VGGFace2
- **Similarity Search**: FAISS (HNSW, LSH, Flat)
- **API**: Flask
- **Image Processing**: PIL, OpenCV

## Author

Patrick Bruce

## License

This project is for educational and portfolio purposes.
