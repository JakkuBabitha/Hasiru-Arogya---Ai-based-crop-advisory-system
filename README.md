# 🌾 AI Crop Advisory System — Hasiru Arogya
### ಹಸಿರು ಆರೋಗ್ಯ | Smart Farming for Karnataka Farmers

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Flask-2.x-000000?style=for-the-badge&logo=flask&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Whisper-OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white"/>
  <img src="https://img.shields.io/badge/Language-Kannada%20%7C%20English-green?style=for-the-badge"/>
</p>

An intelligent, bilingual (Kannada + English) crop advisory web application built for Karnataka farmers.  
The system uses **Machine Learning** to recommend the best crop based on soil conditions and **Deep Learning (CNN)** to detect plant diseases from leaf photos — all accessible via **voice or text** in both Kannada and English.

---

## 📋 Table of Contents
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Modules](#-modules)
- [Installation](#-installation)
- [Running the Application](#-running-the-application)
- [Voice Input Support](#-voice-input-support)
- [Kannada Dialect Support](#-kannada-dialect-support)
- [Dataset](#-dataset)
- [Model Performance](#-model-performance)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🌱 **Crop Recommendation** | Recommends the best crop using soil & climate data (N, P, K, pH, Rainfall, etc.) |
| 🔬 **Disease Detection** | Identifies plant diseases from a leaf photo using a CNN model |
| 🎙️ **Voice Input (Kannada + English)** | Speak your query — the system transcribes and processes it automatically |
| 🌐 **Offline Voice** | Works without internet using OpenAI Whisper (runs locally on CPU) |
| 📖 **Bilingual UI** | Full interface in Kannada (ಕನ್ನಡ) and English |
| 🗺️ **Dialect Support** | Understands all major Karnataka dialects (North, South, Coastal, Mysore) |
| 💊 **Remedy Database** | Provides detailed, actionable remedies for detected diseases |
| 🧠 **Smart NLP Engine** | Extracts crop names, disease symptoms, and soil values from free-form natural language |

---

## 🏗️ System Architecture

```
Farmer Input
     │
     ├── 🎙️ Voice (Online)   → Browser Web Speech API → NLP Engine
     │
     ├── 🎙️ Voice (Offline)  → OpenAI Whisper (local) → Phonetic Correction → NLP Engine
     │
     ├── ⌨️ Text             → NLP Engine
     │
     └── 📷 Leaf Photo       → MobileNetV2 CNN → Disease Label → Remedy Database
                                                                        │
                                              ┌─────────────────────────┘
                                              │
                                    Module 1 (Soil Input)
                                    Random Forest Classifier
                                    → Crop Recommendation
```

---

## 🛠️ Tech Stack

**Backend**
- **Python 3.9+** — Core language
- **Flask** — Web server & REST API
- **Scikit-learn** — Random Forest model (Module 1)
- **TensorFlow / Keras** — MobileNetV2 CNN model (Module 2)
- **OpenAI Whisper** — Offline speech-to-text (supports Kannada + English)
- **ffmpeg** — Audio processing for Whisper

**Frontend**
- **HTML5 / CSS3 / Vanilla JS** — No frameworks, fast loading
- **Web Speech API** — Online voice recognition (Chrome)
- **SpeechGrammarList** — Domain-specific vocabulary hinting for agriculture terms

**Data Processing**
- **Pandas / NumPy** — Dataset handling
- **SMOTE (imbalanced-learn)** — Class imbalance correction during training
- **StandardScaler** — Feature normalization

---

## 📁 Project Structure

```
crop_advisory/
│
├── app_final.py              # 🚀 Main Flask application (backend API)
├── module1_train.py          # 🌱 Module 1 training — Random Forest Crop Recommendation
├── module2_train.py          # 🔬 Module 2 training — MobileNetV2 Disease Detection
│
├── templates/
│   └── index.html            # 🖥️ Main frontend UI (bilingual, voice-enabled)
│
├── static/
│   ├── uploads/              # Uploaded leaf images (temporary)
│   └── assets/               # CSS, icons, fonts
│
├── models/
│   ├── random_forest_model.pkl   # Trained Module 1 model
│   ├── label_encoder.pkl         # Crop label encoder
│   ├── scaler.pkl                # Feature scaler
│   └── plant_disease_model.h5    # Trained Module 2 CNN model
│
├── datasets/
│   └── Crop_recommendation.csv   # Soil & climate dataset (Karnataka crops)
│
├── bin/
│   └── ffmpeg.exe                # Local ffmpeg binary for audio processing
│
├── remedies.py               # 💊 Disease remedy database
├── add_dialects.py           # Kannada dialect vocabulary additions
├── add_general_vocab.py      # General Kannada agriculture vocabulary
└── requirements.txt          # Python dependencies
```

---

## 🧩 Modules

### Module 1 — Crop Recommendation (Random Forest)
- **Algorithm:** Random Forest Classifier with **500 Decision Trees**
- **Inputs:** Nitrogen (N), Phosphorus (P), Potassium (K), Temperature, Humidity, pH, Rainfall
- **Output:** Recommended crop from **35+ Karnataka crops**
- **Special Technique:** SMOTE is applied during training to handle class imbalance
- **How it works:** All 500 trees independently vote on the best crop. The majority vote wins, giving a **confidence percentage** along with the recommendation.

### Module 2 — Plant Disease Detection (MobileNetV2 CNN)
- **Architecture:** MobileNetV2 (Transfer Learning from ImageNet)
- **Total Classes:** 159 plant disease categories
- **Input:** A leaf photograph (JPG/PNG)
- **Output:** Detected disease name + confidence score + actionable remedy
- **Why MobileNetV2:** Lightweight, fast on CPU, and highly accurate — ideal for field use on basic hardware.

### Smart NLP Engine (`process_smart_query()`)
- Extracts crop names, disease symptoms, and soil values from **free-form voice or text input**
- Supports **Kannada Kannada** (30+ crop names, 20+ disease terms) mapped to English equivalents
- Works with mixed Kannada-English sentences ("My ಟೊಮ್ಯಾಟೊ plant has yellowing")

---

## ⚙️ Installation

### Prerequisites
- Python 3.9 or higher
- pip
- Git
- A microphone (for voice input features)

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/crop_advisory.git
cd crop_advisory
```

### 2. Create & Activate Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Whisper Model (First Run Only)
The Whisper `small` model (~460MB) downloads automatically on the first server startup. Ensure you have an internet connection for the first run.

### 5. Place Model Files
Ensure the following trained model files exist in the `models/` folder:
```
models/
├── random_forest_model.pkl
├── label_encoder.pkl
├── scaler.pkl
└── plant_disease_model.h5
```
> **Note:** If model files are missing, run `module1_train.py` and `module2_train.py` to train them.

### 6. Train Models (Optional — if .pkl/.h5 files are missing)
```bash
# Train Module 1 (Crop Recommendation)
python module1_train.py

# Train Module 2 (Disease Detection)
python module2_train.py
```

---

## 🚀 Running the Application

```bash
python app_final.py
```

Open your browser and go to:
```
http://localhost:5000
```

On startup, the terminal will show:
```
  Module 1 : [READY]
  Module 2 : [READY]
  Whisper  : [READY]
  Open browser: http://localhost:5000
```

---

## 🎙️ Voice Input Support

The system has two voice modes:

| Mode | Engine | Internet Required | Languages |
|------|--------|------------------|-----------|
| **Online** | Browser Web Speech API | ✅ Yes | Kannada (kn-IN), English (en-IN) |
| **Offline** | OpenAI Whisper (local) | ❌ No | Kannada + English (auto-detected) |

### How Offline Voice Works
1. Farmer speaks into the microphone.
2. Audio is captured in the browser and sent to the Flask server.
3. **Whisper** transcribes it locally on the CPU (no internet, no API key needed).
4. The **Dual-Language Pass** runs transcription twice (Kannada + English) and picks the one with higher confidence.
5. **Phonetic Correction** fixes common mishearings (e.g., "torpedo" → "tomato", "bright" → "blight").
6. The corrected text is displayed in an **editable confirmation box** so the farmer can fix any error before submitting.

---

## 🗺️ Kannada Dialect Support

The NLP engine understands all major Karnataka regional dialects:

| Region | Local Crop Words Understood |
|--------|-----------------------------|
| **North Karnataka** (Dharwad/Belagavi) | ಉಳ್ಳಾಗಡ್ಡಿ (Onion), ಮೆಕ್ಕೆಜೋಳ (Corn) |
| **Coastal Karnataka** (Mangaluru/Udupi) | ಅಡಿಕೆ (Arecanut), ಭತ್ತ (Paddy) |
| **South Karnataka** (Mysuru/Mandya) | ಕಬ್ಬು (Sugarcane), ಬಾಳೆ (Banana) |
| **Standard Kannada** (Bengaluru) | ಟೊಮ್ಯಾಟೊ (Tomato), ಈರುಳ್ಳಿ (Onion) |

All local Kannada crop and disease words are internally mapped to English labels before being sent to the ML models.

---

## 📊 Dataset

**Module 1 — Crop Recommendation**
- **Source:** Karnataka Crop Recommendation Dataset
- **Samples:** 2200+ rows
- **Features:** N, P, K, Temperature, Humidity, pH, Rainfall
- **Labels:** 35+ crops

**Module 2 — Disease Detection**
- **Source:** PlantVillage Dataset (extended)
- **Total Classes:** 159 plant disease categories
- **Image Format:** JPG, 224×224 pixels

---

## 📈 Model Performance

### Module 1 — Random Forest (Crop Recommendation)
| Metric | Score |
|--------|-------|
| Accuracy | ~99% |
| Trees | 500 |
| Train/Test Split | 70% / 30% |
| Class Balance | SMOTE applied |

### Module 2 — MobileNetV2 (Disease Detection)
| Metric | Score |
|--------|-------|
| Architecture | MobileNetV2 (Transfer Learning) |
| Disease Classes | 159 |
| Input Size | 224 × 224 |

---

## 👨‍💻 Author

Built with ❤️ for Karnataka farmers.

---

## 📄 License

This project is for educational and research purposes. Feel free to fork and build upon it for agricultural applications.
