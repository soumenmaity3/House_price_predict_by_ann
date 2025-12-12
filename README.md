# 🏠 House Rent Prediction App  
### Deep Learning • ANN Regression • Streamlit • Smart Locality Auto-Correction

Welcome to the **House Rent Prediction System**, an end-to-end Machine Learning + Deep Learning project that predicts rental prices for houses across major Indian cities.  
This project includes **dataset preprocessing**, **ANN model training**, and a fully interactive **Streamlit app** with **AI-powered locality correction**.

---

# 📌 Table of Contents
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

# 🚀 Features

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

# 📁 Project Structure

