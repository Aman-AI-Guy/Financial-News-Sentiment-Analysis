# Financial News Sentiment Analysis

This notebook labels sentences from financial news as neutral, positive, or negative. It compares two PyTorch models trained from scratch:

1. **Baseline**: averages the word embeddings of a sentence and feeds the result to a linear classifier.
2. **Bi-LSTM with attention**: a bidirectional LSTM with an attention layer. The attention weights show which words the model focused on.

The baseline won. It reached 74.9% test accuracy against 69.3% for the Bi-LSTM.

Everything is in [`financial_news_sentiment.ipynb`](financial_news_sentiment.ipynb).

## Dataset

The data is the [Financial PhraseBank](https://arxiv.org/abs/1307.5336) (Malo et al., 2014): 4,846 English sentences from financial news. Each sentence carries the label that at least 50% of the annotators agreed on.

| Label | Sentences |
|---|---:|
| neutral | 2,879 |
| positive | 1,363 |
| negative | 604 |

The notebook loads the CSV from [ukairia777/finance_sentiment_corpus](https://github.com/ukairia777/finance_sentiment_corpus), so it needs internet access. `Sentences_50Agree.txt` in this repo is the same set of 4,846 sentences in the original PhraseBank format (`sentence@label`). The notebook doesn't read it; it's kept here as an offline copy.

## Method

- **Split:** 80/10/10 train/val/test (3,876 / 485 / 485), stratified by label.
- **Tokenizer:** a regex tokenizer that handles financial text. Currency amounts become `<MONEY>` (for example `EUR0 .05`), percentages become `<PERCENT>` (`5.4-per-cent`, `30 %`), and other numbers become `<NUM>`. The vocabulary is built from the training set and keeps only words seen at least twice, giving 4,116 tokens.
- **Class imbalance:** the cross-entropy loss is weighted by inverse class frequency (0.38 neutral, 0.80 positive, 1.82 negative).
- **Training:** Adam, gradient clipping at 1.0, ReduceLROnPlateau, and early stopping on validation loss with patience 3. The checkpoint with the best validation loss is restored before testing.
- **Baseline:** 100-dim embeddings, masked mean pooling, dropout 0.3, and a linear layer.
- **Bi-LSTM with attention:** 64-dim embeddings, a single-layer BiLSTM with 48 hidden units per direction, and additive attention with padding masked out. It uses dropout 0.5 and label smoothing 0.05.

## Results

| Model | Best val loss | Best val accuracy | Test accuracy | Test macro F1 |
|---|---:|---:|---:|---:|
| Embedding average pooling | 0.6208 | 0.7670 | **0.7485** | **0.68** |
| Bi-LSTM + attention | 0.7989 | 0.7588 | 0.6928 | 0.62 |

Test F1 by class:

| Class | Baseline | Bi-LSTM + attention |
|---|---:|---:|
| neutral | 0.84 | 0.79 |
| positive | 0.67 | 0.59 |
| negative | 0.54 | 0.49 |

The two models were close on validation accuracy. On the test set, the Bi-LSTM fell 5.6 points behind the baseline. Most sentences are short and carry their sentiment in a few words ("loss", "rise", "profit"). An average of word embeddings picks those up, while the LSTM has more parameters than 3,876 training sentences can support. Negative is the weakest class for both models, since it has the fewest examples.

### Attention heatmaps

The notebook's last section runs six held-out test sentences through the attention model and plots the attention weight of each token. For "Earnings per share (EPS) amounted to a loss of EUR0.05" the model predicts negative with 82% confidence, and `<MONEY>` and `loss` are among the tokens with the highest weights. On longer sentences, the attention often lands on filler words such as "has", "to", and "said". So attention weights show where the model is looking, but they don't reliably explain why it made a prediction.

## Running it

```bash
pip install torch pandas numpy scikit-learn matplotlib
jupyter notebook financial_news_sentiment.ipynb
```

Run the cells in order in a fresh kernel. Training takes a few minutes on CPU. The seed is fixed at 42, and deterministic cuDNN is turned on when CUDA is available. Results on a GPU may differ slightly.

The results above were recorded on CPU with Python 3.12.3, PyTorch 2.11.0, pandas 3.0.2, NumPy 2.4.4, scikit-learn 1.8.0, and Matplotlib 3.10.9.

## Citation

```bibtex
@article{Malo2014GoodDO,
  title={Good debt or bad debt: Detecting semantic orientations in economic texts},
  author={P. Malo and A. Sinha and P. Korhonen and J. Wallenius and P. Takala},
  journal={Journal of the Association for Information Science and Technology},
  year={2014},
  volume={65}
}
```
