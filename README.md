# 🐚 Abalone Age Classification using Computational Intelligence

Advanced Computational Intelligence framework for Abalone age classification using Feedforward Neural Networks (FNN), Neuro-Fuzzy Systems, Self-Constructing Neuro-Fuzzy Inference System (SONFIN), XGBoost, LightGBM, CatBoost, and Stacking Ensemble learning.

This project focuses on:

- Neural network optimisation
- Computational intelligence techniques
- Fuzzy reasoning interpretability
- Self-constructing fuzzy rule generation
- Ensemble learning
- Real-world deployment with Streamlit

---

# 📌 Project Overview

The goal of this project is to classify Abalone age into multiple ordered classes using physical measurements and computational intelligence approaches.

The project investigates:

- Baseline Feedforward Neural Networks
- Advanced Deep Neural Networks
- Fixed Neuro-Fuzzy Systems
- Gradient Boosting Models
- Stacking Ensemble Learning
- Self-Constructing Neuro-Fuzzy Inference System (SONFIN)

The implementation compares predictive performance, interpretability, fuzzy reasoning capability, and generalisation performance on hidden Kaggle evaluation data.

---

# 🧠 Models Implemented

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

# 🚀 Key Features

✅ Feedforward Neural Networks (FNN)

✅ Self-Constructing Neuro-Fuzzy Inference System (SONFIN)

✅ Dynamic fuzzy rule generation

✅ Gaussian membership functions

✅ Ensemble learning

✅ Stacking meta-learning

✅ Explainable AI visualisations

✅ Kaggle leaderboard evaluation

✅ Streamlit deployment ready

---

# 📊 Experimental Highlights

| Metric | Best Result |
|---|---|
| Best Kaggle Score | 0.808 |
| Best Standalone Model | XGBoost |
| Best Ensemble Model | Stacking Ensemble |
| Best Interpretability Model | SONFIN |

---

# 🧬 SONFIN Implementation

The project includes a Self-Constructing Neuro-Fuzzy Inference System (SONFIN) that:

- Starts without predefined fuzzy rules
- Dynamically generates rules during training
- Learns Gaussian membership functions
- Performs adaptive structure learning
- Combines fuzzy reasoning with neural optimisation

This enables both:

- Predictive modelling
- Explainable computational intelligence reasoning

---

# 📁 Project Structure

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
│   └── sonfin_model.pth
│
├── notebook/
│   └── Part2_Final.ipynb
│
└── data/
    └── sample_input.csv
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/abalone-age-classification-sonfin.git
cd abalone-age-classification-sonfin
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Streamlit Application

```bash
streamlit run app.py
```

---

# 🖥️ Streamlit Demonstration

The Streamlit application allows users to:

- Enter Abalone measurements
- Predict age class in real time
- Visualise model predictions
- Demonstrate fuzzy reasoning behaviour

Predicted Classes:

| Class | Meaning |
|---|---|
| 0 | Young Abalone |
| 1 | Medium Age Abalone |
| 2 | Old Abalone |

---

# 📈 Visualisations Included

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
- Ensemble Learning
- Explainable AI
- Adaptive Fuzzy Rule Learning
- Real-world ML Deployment

---

# 📌 Kaggle Performance

The stacking ensemble achieved the best hidden-test generalisation performance with a Kaggle leaderboard score of:

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
