# LLM Generated Text Detection

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/Deep%20Learning-TensorFlow-orange.svg)
![Keras](https://img.shields.io/badge/Keras-API-red.svg)
![NLP](https://img.shields.io/badge/Domain-NLP-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Table of Contents

- Overview
- Problem Statement
- Project Highlights
- Repository Structure
- Dataset
- Technology Stack
- Machine Learning Workflow
- Data Preprocessing
- Model Architecture
- Model Training
- Results
- Installation
- Future Improvements
- Learning Outcomes
- Author
- License


## 📌Overview

LLM Generated Text Detection is a Natural Language Processing (NLP) and Deep Learning project developed to distinguish **AI-generated essays** from **human-written essays**.

With the rapid advancement of Large Language Models (LLMs), identifying AI-generated content has become increasingly important in education, research, publishing, and digital communication. This project demonstrates an end-to-end machine learning workflow for text classification using an LSTM-based neural network implemented with TensorFlow/Keras.

The project covers the complete pipeline including:

- Data preprocessing
- Text tokenization
- Sequence padding
- Deep Learning model development
- Model training
- Model evaluation
- Prediction on unseen data

---

# 🎯Problem Statement

Recent advancements in generative AI have made it increasingly difficult to distinguish between machine-generated and human-written content.

The objective of this project is to develop a deep learning model capable of accurately classifying text into one of two categories:

- Human Written
- AI Generated

Such systems can assist educational institutions, publishers, recruiters and content verification platforms in identifying synthetic text.

---

#  Project Highlights

- End-to-End NLP Pipeline
- Deep Learning based Text Classification
- LSTM Neural Network
- TensorFlow / Keras Implementation
- Sequence Tokenization
- Text Vectorization
- Model Evaluation
- Prediction Pipeline

---

#  📂Repository Structure

```
LLM-Generated-Text-Detection-Model
│
├── data/
│   ├── AI ESSAY.xlsx
│   ├── train_essays.csv
│   ├── train_prompts.csv
│   ├── test_essays.csv
│   └── sample_submission.csv
│
├── notebooks/
│   └── AI GENERATED TEXT DETECTION.ipynb
│
├── models/
│   └── model.h5
│
├── docs/
│   └── Project Presentation
│
├── LICENSE
├── README.md
└── requirements.txt
```

---

# Dataset

The project uses a dataset containing essays written by both humans and AI models.

Dataset Components

- Training Essays
- Test Essays
- Prompt Dataset
- Submission Template
- Additional AI Essay Dataset

Target Classes

- Human Written
- AI Generated

---

# Technology Stack

Programming Language

- Python

Libraries

- TensorFlow
- Keras
- NumPy
- Pandas
- Scikit-Learn
- Matplotlib
- Seaborn
- NLTK

Development Environment

- Jupyter Notebook

---

# Machine Learning Workflow

```
Dataset Collection
        │
        ▼
Data Cleaning
        │
        ▼
Text Tokenization
        │
        ▼
Sequence Encoding
        │
        ▼
Padding
        │
        ▼
Train-Test Split
        │
        ▼
LSTM Model Training
        │
        ▼
Model Evaluation
        │
        ▼
Prediction
```

---

# Data Preprocessing

The notebook performs several preprocessing operations before model training.

- Dataset loading
- Tokenization
- Vocabulary creation
- Integer sequence encoding
- Sequence padding
- Dataset splitting into training and testing sets

---

# 🤖Model Architecture

The classification model is implemented using TensorFlow/Keras.

Architecture includes:

- Embedding Layer
- Multiple LSTM Layers
- Dense Output Layer
- Sigmoid Activation

The model learns contextual information from sequential text data to classify whether an essay is AI-generated or human-written.

---

# Model Training

The notebook includes:

- Tokenizer fitting
- Sequence conversion
- Padding
- Train/Test Split
- Model compilation
- Model training
- Performance evaluation
- Prediction generation

---

# 📈Results

The trained model successfully classifies essays into:

- Human Written
- AI Generated

The notebook includes evaluation using classification accuracy and prediction outputs.

> **Note:** Reported performance metrics may vary depending on dataset version, train-test split, and training configuration.

---

# 🔮Future Improvements

Possible enhancements include:

- Fine-tuning Transformer Models (BERT, RoBERTa)
- Hyperparameter Optimization
- Cross Validation
- Explainable AI (SHAP/LIME)
- Attention Mechanisms
- Streamlit Web Application
- REST API Deployment
- Docker Containerization
- Cloud Deployment
- Model Monitoring

---

# 🚀Installation

Clone the repository

```bash
git clone https://github.com/yadavraje/LLM-Generated-Text-Detection-Model.git
```

Navigate into the project

```bash
cd LLM-Generated-Text-Detection-Model
```

Install dependencies

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook

```bash
jupyter notebook
```

Open

```
notebooks/AI GENERATED TEXT DETECTION.ipynb
```

---

# Project Status

Current Status

✅ Completed Academic / Portfolio Project

Future Version

- Modular Python Package
- Improved Deep Learning Models
- Transformer-based NLP Models
- Production-ready Deployment

---

# Learning Outcomes

This project demonstrates practical knowledge of:

- Natural Language Processing
- Deep Learning
- Sequence Modeling
- TensorFlow
- Keras
- Data Preprocessing
- Machine Learning Workflow
- Model Evaluation

---

# Author

**Rajesh Yadav**

Mechanical Engineer | Project Manager | AI & Machine Learning Enthusiast

Areas of Interest

- Artificial Intelligence
- Machine Learning
- NLP
- Industrial Automation
- Process Engineering
- Engineering Software Development

---

# License

This project is licensed under the MIT License.

See the LICENSE file for details.