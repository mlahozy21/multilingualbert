# Structural Probes for Spanish Syntax: mBERT & AnCora

This repository contains an experimental replication of the paper **"A Structural Probe for Finding Syntax in Word Representations"** (Hewitt & Manning, 2019), adapted specifically for the **Spanish language**.

The primary objective is to investigate whether multilingual language models (such as **mBERT**) encode Spanish syntactic structure (dependency trees) within their vector geometry, without having been explicitly trained with syntactic supervision.

## 📊 Results

A linear probe was trained for the **Parse Distance** task using embeddings from the 12th layer of `bert-base-multilingual-cased`.

| Metric | Result (Spanish) | Interpretation |
| :--- | :---: | :--- |
| **Spearman Correlation** | **0.735** | **High correlation.** Confirms that mBERT captures the geometric distance between syntactically related words in Spanish. |
| **UUAS** | **0.50** | **Reconstruction accuracy.** Significantly outperforms the random baseline, demonstrating latent structural learning. |

## 🛠️ Modifications & Engineering

To adapt the original 2019 codebase to a modern environment and the specific linguistic features of Spanish, the following key implementations were made:

1.  **Data Alignment (Critical):** Developed a pre-processing script (`conllu_to_text.py`) and modified the data loader (`data.py`) to handle **Spanish contractions** (e.g., *del, al*). A filter was implemented to ignore range indices (e.g., `1-2`) in the CoNLL-U format, which previously caused misalignment between BERT tokens and Universal Dependencies labels.
2.  **Library Migration:** The codebase was updated to remove the obsolete `pytorch-pretrained-bert` dependency and migrate to the modern HuggingFace `transformers` library.
3.  **Mathematical Optimization:** The distance calculation algorithm in `task.py` was rewritten. The original iterative implementation was replaced with a **vectorized Floyd-Warshall algorithm** using NumPy. This eliminated infinite loops caused by cycles in the data and reduced pre-computation time from minutes to seconds.
4.  **Windows/UTF-8 Compatibility:** Forced `utf-8` encoding for all file read/write operations to ensure correct processing of Spanish accents and special characters on Windows systems.

## 📂 Project Structure

```text
structural-probes/
├── data/
│   └── es_ancora/
│       ├── es_ancora-ud-train.conllu  # Original Dataset (UD)
│       ├── es_ancora-ud-train.txt     # Cleaned raw text (ranges removed)
│       └── es_ancora-ud-train.hdf5    # Pre-computed mBERT embeddings
├── example/
│   └── config/
│       └── es_ancora.yaml             # Experiment configuration
├── structural-probes/                 # Modified source code
│   ├── data.py                        # Data loader with contraction filter
│   ├── task.py                        # Optimized Floyd-Warshall algorithm
│   └── run_experiment.py              # Main execution script
├── conllu_to_text.py                  # Custom script for cleaning and extraction
└── generate_embeddings.py             # Custom script to generate mBERT vectors
