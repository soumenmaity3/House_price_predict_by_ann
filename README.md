# 🏠 House Rent Prediction App  
### Deep Learning • ANN Regression • Streamlit • Smart Locality Auto-Correction

Welcome to the **House Rent Prediction System**, an end-to-end Machine Learning + Deep Learning project that predicts rental prices for houses across major Indian cities. This project includes **dataset preprocessing**, **ANN model training**, and a fully interactive **Streamlit app** with **AI-powered locality correction**.

---

## 📌 Table of Contents
- [🚀 Features](#-features)
- [📁 Project Structure](#-project-structure)
- [⚙️ Installation & Setup](#️-installation--setup)
- [🧠 Model Architecture](#-model-architecture)
- [📊 Workflow Overview](#-workflow-overview)
- [🧪 Sample Prediction Output](#-sample-prediction-output)
- [📈 Model Performance](#-model-performance)
- [🛠 Technologies Used](#-technologies-used)
- [💡 Future Enhancements](#-future-enhancements)
- [🤝 Contributing](#-contributing)
- [📬 Contact](#-contact)

---

## 🚀 Features

### ✔️ **1. Deep Learning Model (ANN)**
- Built using **TensorFlow / Keras Functional API**
- Locality handled using **Embedding layer** (supports 2000+ areas)
- Dense architecture with **BatchNorm** + **Dropout**
- Predicts **log(rent)** for better model stability

### ✔️ **2. Intelligent Streamlit Web Application**
- Clean UI for entering house details
- Fully responsive design
- Predicts rent in real time

### ✔️ **3. Smart Locality Auto-Correction (AI Fuzzy Matching)**
If a user types a wrong locality, the app suggests the closest match:

```
"Whitefiled" → Whitefield?
"Bandal" → Bandel?
"Kormangla" → Koramangala?
```

Uses **RapidFuzz** for robust fuzzy search.

### ✔️ **4. Realistic Input Validation**
Detects incorrect or unrealistic values:
- Size too small for BHK
- Too many bathrooms
- Floors exceeding total floors
- Extremely large or tiny apartments

### ✔️ **5. Preprocessor & Model Files Included**
Saved using pickle & Keras:
- `model.h5`
- `lb_encoder.pkl`
- `one_encoder.pkl`
- `scaler.pkl`

---

## 📁 Project Structure

```
House-Rent-Prediction/
│
├── Notebook/
│   ├── data_cleaning.ipynb
│   ├── model_training.ipynb
│   ├── app.py
│   ├── model.h5
│   ├── lb_encoder.pkl
│   ├── one_encoder.pkl
│   └── scaler.pkl
│
├── Data/
│   └── House_Rent_Dataset.csv
│
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation & Setup

### **1️⃣ Clone Repository**
```bash
git clone https://github.com/yourusername/house-rent-prediction.git
cd house-rent-prediction/Notebook
```

### **2️⃣ Create Virtual Environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### **3️⃣ Install Required Packages**
```bash
pip install -r ../requirements.txt
```

### **4️⃣ Run App**
```bash
streamlit run app.py
```

---

## 🧠 Model Architecture

```
Inputs:
  • Locality ID → Embedding(32) → Flatten
  • Other Features (scaled numerics + one-hot encoded)

Merged → Dense(256) → BatchNorm → Dropout
       → Dense(128) → BatchNorm → Dropout
       → Dense(64)

Output:
       → Dense(1) (predicts log(rent))
```

**Target transformation used:**

```python
log_rent = np.log1p(rent)  
predicted_rent = np.expm1(log_output)
```

---

## 📊 Workflow Overview

### **1. Dataset Cleaning**
- Splitting `Floor` into `Current_Floor` & `Total_Floors`
- Encoding tenant preferences (`bachelor`, `family`)
- One-hot encoding:
  - Area Type
  - City
  - Furnishing Status
- Label encoding:
  - Locality (as `Locality_ID`)
- Handling missing values
- Feature scaling with StandardScaler
- Creating target feature `LogRent`

### **2. Model Training**
- ANN model trained on log-transformed rent
- Early stopping
- Validation split
- Saved trained model as `model.h5`

### **3. Streamlit Deployment**
- Handles real-time predictions
- Intelligent locality correction
- Full input validation

---

## 🧪 Sample Prediction Output

Example:

```
================ NEW PREDICTION REQUEST ================
City:               Bangalore
Locality:           Whitefield (ID: 2141)
BHK:                2
Size:               1000 sqft
Bathrooms:          2
Current Floor:      3
Total Floors:       10
Bachelor Allowed:   0
Family Allowed:     1
Area Type:          Super Area
Furnishing:         Semi-Furnished
-------------------------------------------------------
💰 PREDICTED RENT: ₹ 13,382.28
========================================================
```

---

## 📈 Model Performance

| Metric       | Score           |
| ------------ | --------------- |
| **MAE**      | 11,000 – 13,000 |
| **RMSE**     | 30,000 – 35,000 |
| **R² Score** | ~0.82           |

---

## 🛠 Technologies Used

| Component            | Library                     |
| -------------------- | --------------------------- |
| Framework            | TensorFlow, Keras           |
| Backend              | Python                      |
| Deployment           | Streamlit                   |
| Preprocessing        | Pandas, NumPy, Scikit-Learn |
| Locality Suggestions | RapidFuzz                   |
| Visualization        | Streamlit UI                |

---

## 💡 Future Enhancements

- Add **SHAP explainability**
- Add **interactive city maps**
- Deploy as **REST API (FastAPI)**
- Add **historical rent insights**
- Add **model retraining pipeline**

---

## 🤝 Contributing

Contributions, pull requests, and suggestions are welcome!

To contribute:

1. Fork the repo
2. Create your feature branch
3. Submit a pull request

---

## ⭐ Support This Project

If this project helped you, please **star ⭐ the repository** — it encourages future improvements!

---

## 📬 Contact

For questions, suggestions, or collaboration:

**Email:** yourname@gmail.com

**GitHub:** [github.com/yourusername](https://github.com/yourusername)

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

*Made with ❤️ for the Data Science Community*