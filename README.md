# CSL 7640 — Natural Language Understanding | Assignment-2

**Author:** Vishal | **Roll No:** B23CM1048

---

## Repository Structure

```
ASSIGNMENT-2/
│
├── Problem 1/                        # Word Embeddings on IITJ Corpus
│   ├── data/
│   │   ├── About.txt
│   │   ├── AnnualReport_2024-25_IITJ.pdf
│   │   ├── Btech_regulations_IITJ.pdf
│   │   ├── IITJ_Academic_Regulations.pdf
│   │   ├── Departments.txt
│   │   ├── Programs.txt
│   │   └── urls.txt                  # List of IITJ URLs to scrape
│   ├── outputs/                      # Generated files (corpus, models, plots)
│   ├── preprocess_pipeline.py        # Data collection, cleaning, corpus builder
│   ├── train_word2vec.py             # Train CBOW & Skip-gram (3 configs each)
│   ├── semantic_analysis.py          # Nearest neighbors + analogy experiments
│   ├── visualize_embeddings.py       # PCA & t-SNE plots for both models
│   └── wordcloud_gen.py              # Word cloud from corpus
│
├── problem 2/                        # Character-Level Name Generation
│   ├── models/
│   │   ├── rnn_vanilla.py            # Vanilla RNN (from scratch)
│   │   ├── blstm_model.py            # Bidirectional LSTM (from scratch)
│   │   └── rnn_attention.py          # RNN + Bahdanau Attention (from scratch)
│   ├── outputs/                      # Checkpoints, generated names, plots
│   ├── TrainingNames.txt             # 1000 Indian names dataset
│   ├── utils.py                      # CharVocab, Dataset, collate_fn, metrics
│   ├── train.py                      # Train all three models
│   ├── generate.py                   # Sample names from trained models
│   ├── evaluate.py                   # Novelty rate & diversity score
│   └── visualize.py                  # Loss curves, metrics bar chart, heatmap
│
├── nlu_env/                          # Python virtual environment
├── .gitignore
└── README.md
```

---

## Setup

```bash
# Create and activate virtual environment
python -m venv nlu_env
nlu_env\Scripts\activate        # Windows
# source nlu_env/bin/activate   # Linux/Mac

# Install dependencies
pip install gensim nltk pdfplumber requests beautifulsoup4
pip install wordcloud matplotlib scikit-learn
pip install torch
```

---

## Problem 1 — Running Order

```bash
cd "Problem 1"

# Step 1: Build corpus from PDFs, TXTs, and URLs
python preprocess_pipeline.py

# Step 2: Train CBOW & Skip-gram across 3 hyperparameter configs
python train_word2vec.py

# Step 3: Nearest neighbors + analogy experiments
python semantic_analysis.py

# Step 4: PCA & t-SNE visualizations
python visualize_embeddings.py

# Step 5: Word cloud
python wordcloud_gen.py
```

All outputs (corpus, models, plots) are saved to `Problem 1/outputs/`.

---

## Problem 2 — Running Order

```bash
cd "problem 2"

# Step 1: Train Vanilla RNN, BLSTM, RNN+Attention
python train.py

# Step 2: Generate 500 names per model
python generate.py

# Step 3: Compute novelty rate & diversity score
python evaluate.py

# Step 4: Generate all plots (loss curves, metrics, heatmap)
python visualize.py
```

All outputs are saved to `problem 2/outputs/`.

---

## Key Results

| | Problem 1 | Problem 2 |
|---|---|---|
| **Best model** | Skip-gram (quality 0.364) | Vanilla RNN (diversity 98.8%) |
| **Corpus/Data** | 11,069 sentences, 14,573 vocab | 1,000 Indian names |
| **Notable finding** | Skip-gram outperforms CBOW on all configs | BLSTM overfits severely (loss → 0.01) |