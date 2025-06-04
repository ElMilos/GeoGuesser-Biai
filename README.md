# 🌍Guessing geographical location based on landscape image – Biologically Inspired AI for Geolocation

A deep learning project that aims to **predict the country of origin of a landscape photo** using image recognition techniques inspired by biological systems.

> 📚 Developed as part of a university course in Biologically Inspired Artificial Intelligence (BIAI) by  
> Eustachy Lisiński & Miłosz Wojtanek

---

## 📌 Project Description

Using deep learning models trained on real-world landscape photos, this project attempts to classify the country shown in an image.
In-depth report available in repository in file named "BIAI - Raport-final"
---

## 🧠 Technologies & Tools Used

- **Python 3.10**
- **TensorFlow / Keras** – model building and training
- **Matplotlib** – training visualization
- **Anaconda** – environment and dependency management
- **CNN, LSTM, and hybrid CNN+LSTM architectures**

---

## 🗂 Dataset

- **Source**: [GeoGuessr AI Dataset on Roboflow](https://universe.roboflow.com/geoguessr-1st-dataset/geoguessr-ai-1/dataset/4)
- **Size**: ~18,000 images from 8 countries
- **Split**: 70% training / 20% validation / 10% test
- **Format**: Images organized into folders per country (classification)

---

## 🛠 Repository Structure

| File | Description |
|------|-------------|
| `TrainNew.py` | Trains a new CNN or hybrid model from scratch |
| `TrainLoaded.py` | Further trains an existing model |
| `TestLoaded.py` | Loads a model and classifies a given image |
| `/models/` | Directory for saved `.keras` models |
| `/Data/` | Directory for training and test images |

---

## 🚀 How to Run

> Requirements: TensorFlow, Keras, Matplotlib, Pillow, NumPy

```bash
# Train a new model
python TrainNew.py --save_path models/my_model.keras

# Continue training an existing model
python TrainLoaded.py --model_name models/my_model.keras

# Run prediction on a single image
python TestLoaded.py --model_name models/my_model.keras --img_path Data/example.jpg
```
## 📊 Results
Final CNN+LSTM hybrid architecture reached up to 80% test accuracy

Experiments included:

CNN-only models (baseline)

Overfitting countermeasures: L2 regularization, dropout, batch normalization

LSTM and hybrid CNN+LSTM model comparisons

Multiple optimizers: Adam, AdaMax, AdaDelta
