# Multimodal Product Classification

This project focuses on classifying BestBuy.com products into predefined categories using both textual and visual data. It combines natural language processing and computer vision to build robust machine learning models for multi-class classification.

##  Project Overview

Each product has:
- Name and description (text)
- Image (URL)
- Metadata (price, type, manufacturer, etc.)

We aim to:
- Generate embeddings from product descriptions and images
- Train and evaluate models using both modalities (text, image, or both)
- Achieve high accuracy and F1-score across categories

##  Dataset

The dataset includes:
- `processed_products_with_images.csv`: preprocessed product metadata
- Product images: JPEG files (224x224)
- Optional: `categories.json` for hierarchy

Each product includes:
- `name`, `description`, `image`, `price`, `class_id`, etc.

##  Tech Stack

- **Python**
- **Pandas & NumPy**: Data manipulation
- **TensorFlow / Keras**: Deep learning
- **Scikit-learn**: Classic ML models
- **HuggingFace Transformers**: Text embeddings (e.g. MiniLM)
- **ConvNextV2 / ResNet50**: Image embeddings
- **Matplotlib / Seaborn**: Visualization
- **Pytest**: Testing
- **Docker**: Optional containerization

##  Models

- Text-only models: Logistic Regression, Random Forest, MLP
- Image-only models: Same as above using visual embeddings
- Multimodal models: Early fusion using MLP (text + image)

Minimum performance requirements:
-  Multimodal: 85% accuracy, 80% F1
-  Text-only: 85% accuracy, 80% F1
-  Image-only: 75% accuracy, 70% F1

##  Project Structure

.
├── data/ # Processed dataset and images
├── Embeddings/ # Saved embeddings (CSV)
├── notebooks/ # Project notebook
├── src/ # Core modules (CV, NLP, MLP, utils)
├── tests/ # Unit tests
├── results/ # Evaluation metrics (CSV)
├── Dockerfile # Optional container setup
├── requirements.txt
└── README.md

bash
Copiar
Editar

##  Installation

```bash
git clone https://github.com/ArocenaAlfredo/product-classification-multimodal.git
cd product-classification-multimodal
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
Optional for Mac GPU: pip install -r requirements_mac.txt

 Generate Embeddings
Run the notebook or scripts to:

Download and preprocess images

Generate text embeddings (e.g. MiniLM)

Generate image embeddings (e.g. ConvNextV2, ResNet50)

 Train Models
Run classic ML or deep MLPs for:

Text-only

Image-only

Multimodal (early fusion)

 Run Tests
bash
Copiar
Editar
pytest tests/
 Docker (optional)
bash
Copiar
Editar
docker build -t anyoneai-project .
docker run -p 8888:8888 -v $(pwd):/app anyoneai-project
Then open the link in your browser with the Jupyter token.

This project demonstrates the power of combining modalities for product classification in real-world e-commerce.
