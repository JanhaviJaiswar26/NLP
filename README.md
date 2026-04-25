# Linguistic Anomaly Detection for Speech Disorder Classification

## Overview
A multi-model NLP pipeline for classifying speech disorders 
from transcribed audio, developed as MSc Data Science 
research at SIES College, Mumbai (2024-2025).

## Problem Statement
Automated detection of speech disorders including apraxia, 
dysarthria, stuttering, and normal speech using 
machine learning and transformer-based models.

## Methodology
- Speech-to-Text: Wav2Vec2 (facebook/wav2vec2-base-960h)
- Feature Extraction: TF-IDF vectorization with bigram features
- Models: Random Forest (200 estimators), BERT sequence 
  classification, XGBoost
- Dataset: YouTube speech samples + TIMIT

## Results
- Random Forest: 1.0 accuracy on training evaluation 
  across 4 disorder categories
- BERT: Evaluated on fine-tuning feasibility with 
  limited clinical data
- 4 disorder classes: apraxia, dysarthria, 
  stuttering, normal

## Tools Used
Python, PyTorch, HuggingFace Transformers, 
Wav2Vec2, Scikit-learn, Pandas, Jupyter Notebook

## Research Guide
Dr. Abuzar Ansari, SIES College of Arts, 
Science and Commerce, Mumbai
