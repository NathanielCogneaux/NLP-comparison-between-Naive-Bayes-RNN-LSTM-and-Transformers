# NLP Comparison: Naive Bayes, RNN, LSTM & Transformers for Financial Sentiment Analysis

This repository presents a comparative study of four Natural Language Processing (NLP) architectures:
- Naive Bayes
- Recurrent Neural Networks (RNN)
- Long Short-Term Memory Networks (LSTM)
- Transformers (via BERT)

The analysis is applied to the **Kaggle dataset**: [NLP - Financial News Sentiment Analysis](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset).  
It was conducted as part of a **Master’s degree project** in deep learning.

---

## 📁 Structure

- `models/`
  - `Naive_Bayes.ipynb` – Baseline using scikit-learn's Multinomial Naive Bayes
  - `RNN.ipynb` – Implementation of a vanilla Recurrent Neural Network in PyTorch
  - `LSTM.ipynb` – LSTM-based sentiment classifier
  - `Transformers.ipynb` – Transformer-based model using BERT via Hugging Face

- `NLP_financial_sentiment_Pytorch_Project.pdf`  
  A comprehensive PDF document explaining:
  - Theoretical foundations of each model
  - Pros and cons
  - Real-world use cases
  - Implementation details
  - Experimental results and visualizations

---

## 🧾 Summary of Findings

| Model         | Accuracy | Pros                                      | Cons                                     |
|---------------|----------|-------------------------------------------|------------------------------------------|
| Naive Bayes   | ~70%     | Simple, fast                              | Assumes feature independence             |
| RNN           | ~80%     | Captures sequential information           | Suffers from vanishing gradients         |
| LSTM          | ~81%     | Handles long-term dependencies            | Computationally intensive                |
| Transformers  | ~84%     | High accuracy, parallelizable             | Requires large datasets and more compute |

As expected, more complex models generally achieve better performance at the cost of increased computation.

---

## Getting Started

1. **Clone this repo**  
   ```bash
   git clone https://github.com/yourusername/NLP-comparison-between-Naive-Bayes-RNN-LSTM-and-Transformers.git
   cd NLP-comparison-between-Naive-Bayes-RNN-LSTM-and-Transformers
````

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebooks** in the `models/` folder.

---

## 📖 References

1. Vaswani et al., "Attention is All You Need", NIPS 2017
2. Rumelhart et al., "Learning Representations by Back-Propagating Errors", Nature 1986
3. Hochreiter & Schmidhuber, "Long Short-Term Memory", Neural Computation 1997
4. McCallum & Nigam, "A Comparison of Event Models for Naive Bayes Text Classification", AAAI 1998
5. Goodfellow et al., *Deep Learning*, MIT Press, 2016
6. [Wikipedia: LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory)
7. [La Revue IA: Qu’est-ce qu’un réseau LSTM?](https://larevueia.fr/quest-ce-quun-reseau-lstm/)