# 🐚 Abalone Age Classification using Computational Intelligence

Advanced Computational Intelligence framework for Abalone age classification using Feedforward Neural Networks (FNN), Neuro-Fuzzy Systems, Self-Constructing Neuro-Fuzzy Inference System (SONFIN), XGBoost, LightGBM, CatBoost, and Stacking Ensemble learning.

---

# 🚀 Project Overview

This project investigates multiple Computational Intelligence (CI) approaches for Abalone age classification using physical measurement features.

The implementation focuses on:

- Feedforward Neural Networks (FNN)
- Neuro-Fuzzy Systems
- Self-Constructing Neuro-Fuzzy Inference System (SONFIN)
- Gradient Boosting Models
- Ensemble Learning
- Explainable AI and fuzzy reasoning
- Streamlit deployment for real-time prediction

The project compares model performance, interpretability, fuzzy rule generation capability, and hidden-test generalisation performance.

---

# 🧠 Implemented Models

| Setting | Model |
|---|---|
| 1 | Baseline FNN |
| 2 | Advanced FNN |
| 3 | Fixed Neuro-Fuzzy Model |
| 4 | XGBoost |
| 5 | LightGBM |
| 6 | CatBoost |
| 7 | Stacking Ensemble |
| 8 | Self-Constructing Neuro-Fuzzy Inference System (SONFIN) |

---

# ✨ Key Features

✅ Feedforward Neural Networks (FNN)

✅ Neuro-Fuzzy Inference Systems

✅ Self-Constructing SONFIN architecture

✅ Dynamic fuzzy rule generation

✅ Gaussian membership functions

✅ Ensemble meta-learning

✅ Explainable AI visualisations

✅ Real-time Streamlit deployment

✅ Hidden Kaggle evaluation testing

---

# 📊 Experimental Highlights

| Metric | Result |
|---|---|
| Best Kaggle Score | 0.808 |
| Best Ensemble Model | Stacking Ensemble |
| Best Standalone Model | XGBoost |
| Best Explainability Model | SONFIN |

---

# 🧬 SONFIN Implementation

The project includes a Self-Constructing Neuro-Fuzzy Inference System (SONFIN) that:

- Starts without predefined fuzzy rules
- Dynamically constructs fuzzy rules during training
- Learns Gaussian membership functions adaptively
- Combines fuzzy inference with neural optimisation
- Performs both structure learning and parameter learning

This enables improved interpretability and adaptive computational intelligence reasoning.

---

# 📁 Repository Structure

```text
abalone-age-classification-sonfin/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── xgb_model.pkl
│   ├── lgbm_model.pkl
│   ├── cat_model.pkl
│   ├── meta_model.pkl
│   ├── advanced_fnn.pth
│   └── scaler.pkl
│
└── notebook/
    └── ProjectK.ipynb
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/abalone-age-classification-sonfin.git
cd abalone-age-classification-sonfin
```

Install required libraries:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Streamlit Application

```bash
streamlit run app.py
```

---

# 🖥️ Streamlit Features

The Streamlit application supports:

- Manual user input prediction
- CSV batch prediction upload
- Real-time Abalone age classification
- Prediction confidence visualisation
- Interactive demonstration of ensemble learning

---

# 🐚 Prediction Classes

| Class | Meaning |
|---|---|
| 0 | Young Abalone (Age ≤ 8) |
| 1 | Medium Age Abalone (Age 9–10) |
| 2 | Old Abalone (Age ≥ 11) |

---

# 📈 Included Visualisations

- Learning curves
- Confusion matrices
- Ground truth vs prediction plots
- Fuzzy membership functions
- Feature importance analysis
- Meta-learner importance analysis
- SONFIN rule visualisations

---

# 🛠️ Technologies Used

- Python
- PyTorch
- Scikit-learn
- XGBoost
- LightGBM
- CatBoost
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn

---

# 📚 Academic Focus

This project demonstrates:

- Computational Intelligence (CI)
- Fuzzy Neural Networks
- Neuro-Fuzzy Systems
- Self-Constructing Fuzzy Neural Networks
- Ensemble Learning
- Explainable AI
- Adaptive Rule Learning
- Real-world ML deployment

---

# 📌 Kaggle Performance

The stacking ensemble achieved the strongest hidden-test generalisation performance with a Kaggle leaderboard score of:

```text
0.808
```

---

# 👨‍💻 Author

Sri Sai Hemanth

Master’s Student — Artificial Intelligence

---

# 📄 License

This project is developed for academic and research purposes.
