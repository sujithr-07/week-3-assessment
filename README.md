

---
# Task 1: Neural Network Fundamentals

### Objective

Implement a neural network from scratch using NumPy and understand the fundamental concepts of neural networks.

### Concepts Covered

- Activation functions: ReLU, Sigmoid and Softmax.
- Loss functions: Mean Squared Error and Binary Cross Entropy.
- Forward propagation and backpropagation.
- Gradient descent optimization.
- XOR classification using a 2-4-1 neural network.
- Manual gradient calculation.
- PyTorch Autograd verification.
- Learning rate experiments.
- Loss curve visualization.

### Network Architecture

```text
Input Layer:  2 Neurons
       |
       v
Hidden Layer: 4 Neurons (ReLU)
       |
       v
Output Layer: 1 Neuron (Sigmoid)
```

### Dataset

### Training Configuration

| Parameter     | Value            |
| ------------- | ---------------- |
| Architecture  | 2-4-1            |
| Activation    | ReLU, Sigmoid    |
| Optimizer     | Gradient Descent |
| Epochs        | 5000             |
| Learning Rate | 0.5              |
| Dataset       | XOR              |

### Expected Results

```text
Softmax:
[0.659 0.242 0.099]

XOR Predictions:
[0.02 0.97 0.97 0.03]

Autograd:
w.grad = -32.0
b.grad = -16.0
```

The prediction values are illustrative; actual values depend on initialization and training.

### Learning Rate Experiments

Tested learning rates:

- 0.01 – Slow learning.
- 0.5 – Main training experiment.
- 5 – Potential instability or divergence.

---

# Task 2: PyTorch Training Pipeline

### Objective

Build and train a Multi-Layer Perceptron (MLP) using PyTorch on the Breast Cancer dataset.

### Concepts Covered

- Dataset loading and preprocessing.
- Feature scaling.
- Custom `CsvDataset` class.
- PyTorch DataLoader.
- Neural network implementation using `nn.Module`.
- Linear, ReLU, Dropout and BatchNorm1d layers.
- Adam optimizer.
- Training and validation loops.
- Early stopping.
- Model saving and loading.
- Comparison with Scikit-learn Logistic Regression.

### Dataset

Breast Cancer Wisconsin dataset.

### Model Architecture

```text
Input Features
      |
      v
Linear Layer
      |
      v
BatchNorm1d
      |
      v
ReLU
      |
      v
Dropout
      |
      v
Output Layer
      |
      v
Classification
```

### Training Configuration

| Parameter      | Value                |
| -------------- | -------------------- |
| Framework      | PyTorch              |
| Batch Size     | 32                   |
| Optimizer      | Adam                 |
| Epochs         | 40                   |
| Loss Function  | Binary Cross Entropy |
| Early Stopping | Enabled              |
| Evaluation     | Validation Accuracy  |

### Expected Results

```text
Params: 4,000

Best Validation Accuracy: 0.97
Best Epoch: 28

Reloaded Model Accuracy: 0.97
```

Results may vary depending on the model architecture, random seed and train-validation split.

### Model Comparison

The trained neural network is compared with Scikit-learn Logistic Regression to evaluate classification performance.

---

# Task 3: Classical NLP Baseline and Embeddings

### Objective

Build a classical NLP sentiment classification system using TF-IDF and Logistic Regression, then compare it with GloVe word embeddings.

### Dataset

NLTK Movie Reviews dataset.

- 2,000 labelled movie reviews.
- Positive and negative sentiment classes.
- 80% training data.
- 20% testing data.

### Concepts Covered

- Text preprocessing and cleaning.
- Lowercase conversion.
- Punctuation removal.
- Tokenization.
- TF-IDF feature extraction.
- Logistic Regression classification.
- Accuracy and macro F1 evaluation.
- Confusion matrix.
- Top 10 important words for each class.
- GloVe 50-dimensional embeddings.
- Word similarity.
- Word analogy.
- Sentence vector generation.

### TF-IDF Model

```text
Input Text
    |
    v
Text Cleaning
    |
    v
TF-IDF Vectorization
    |
    v
Logistic Regression
    |
    v
Sentiment Prediction
```

### GloVe Embeddings

Pre-trained embeddings:

```text
glove-wiki-gigaword-50
```

The embeddings are loaded and processed using NumPy without requiring Gensim.

### Word Embedding Experiments

Nearest-word similarity:

- King
- Computer

Word analogy:

```text
king - man + woman = queen
```

Sentence vectors are created by averaging the GloVe vectors of individual words.

### Evaluation Metrics

- Accuracy.
- Macro F1 score.
- Confusion matrix.
- Important words for positive and negative sentiment.

### Model Comparison

| Model                        | Features                |
| ---------------------------- | ----------------------- |
| TF-IDF + Logistic Regression | TF-IDF vectors          |
| GloVe + Logistic Regression  | Average word embeddings |

Both models are evaluated using the same train-test split.

---

# Task 4: Transformers – Pipelines, Fine-tuning and Serving

### Objective

Explore Transformer architectures, use pretrained DistilBERT pipelines, fine-tune a Transformer model and deploy it using FastAPI.

### Model

`distilbert-base-uncased`

### Concepts Covered

- Transformer tokenization.
- Input IDs and attention masks.
- Sentiment analysis pipeline.
- Named Entity Recognition (NER).
- Fine-tuning using Hugging Face Trainer API.
- Model evaluation using accuracy and macro F1.
- Training and inference time measurement.
- TF-IDF vs DistilBERT comparison.
- Model saving and loading.
- FastAPI model deployment.

### 1. Tokenizer

Demonstrates:

- Tokens.
- Input IDs.
- Attention masks.

Two example sentences are used to understand Transformer tokenization.

### 2. Sentiment Analysis

Uses the pretrained DistilBERT sentiment-analysis pipeline to classify three example sentences.

### 3. Named Entity Recognition

Uses a pretrained DistilBERT NER model to identify:

- Person names.
- Organizations.
- Locations.

### 4. Fine-tuning

Fine-tunes DistilBERT using the NLTK Movie Reviews dataset.

| Parameter      | Value                   |
| -------------- | ----------------------- |
| Model          | DistilBERT              |
| Dataset        | Movie Reviews           |
| Training Split | 80%                     |
| Testing Split  | 20%                     |
| Epochs         | 3                       |
| Optimizer      | AdamW (Trainer default) |
| Evaluation     | Accuracy and Macro F1   |

### 5. Model Comparison

Compares TF-IDF and DistilBERT using:

| Metric         | Description                      |
| -------------- | -------------------------------- |
| Accuracy       | Correct predictions              |
| Macro F1       | Average F1 across classes        |
| Training Time  | Total model training time        |
| Inference Time | Average prediction time per text |

The results are saved in:

```text
model_comparison.csv
```

### 6. FastAPI Deployment

Creates a REST API using FastAPI.

Endpoint:

```http
POST /predict
```

Example request:

```json
{
    "text": "This movie is fantastic and wonderful!"
}
```

Example response:

```json
{
    "text": "This movie is fantastic and wonderful!",
    "label": "POSITIVE",
    "confidence": 0.9876
}
```

The confidence value is illustrative.

### API Features

- Loads the trained model once at application startup.
- Accepts text through a POST request.
- Returns sentiment label and confidence.
- Validates empty text.
- Returns HTTP 422 for empty input.

### Run the API

```bash
uvicorn api:app --reload
```

Open Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

---

# Technologies Used

| Technology                | Purpose                       |
| ------------------------- | ----------------------------- |
| Python                    | Programming language          |
| NumPy                     | Numerical computation         |
| Matplotlib                | Data visualization            |
| PyTorch                   | Neural network training       |
| Scikit-learn              | Machine learning models       |
| Pandas                    | Dataset handling              |
| NLTK                      | Text processing and datasets  |
| Hugging Face Transformers | Pretrained Transformer models |
| Hugging Face Datasets     | Dataset processing            |
| FastAPI                   | Model deployment              |
| Uvicorn                   | API server                    |

---

# Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

```bash
cd ML-DL-NLP-Assignments
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

Windows:

```bash
.venv\Scripts\activate
```

Linux / macOS:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run individual tasks

Task 1:

```bash
python Task1/task1.py
```

Task 2:

```bash
python Task2/task2.py
```

Task 3:

```bash
python Task3/task3.py
```

Task 4:

Run the notebook:

```text
Task4/task4.ipynb
```

Start the API from the Task4 directory:

```bash
uvicorn api:app --reload
```

---

# Model and Dataset Downloads

Large model weights and embedding files are not included in this GitHub repository.

They can be downloaded separately:

- DistilBERT: https://huggingface.co/distilbert/distilbert-base-uncased
- GloVe: https://nlp.stanford.edu/data/glove.6B.zip
- NLTK Movie Reviews: Downloaded through NLTK.
- Breast Cancer Dataset: Available through Scikit-learn.

For Task 4, run the fine-tuning notebook before starting the API so that the `saved_distilbert` directory is generated.

---

# Key Learnings

Through these four assignments, I gained practical experience in:

1. Understanding neural network fundamentals and backpropagation.
2. Building custom neural networks using NumPy and PyTorch.
3. Training and evaluating classification models.
4. Understanding classical NLP techniques and word embeddings.
5. Working with Transformer architectures and pretrained models.
6. Fine-tuning DistilBERT for sentiment classification.
7. Comparing classical machine learning and deep learning models.
8. Deploying trained machine learning models using FastAPI.

---
