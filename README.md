# 🍗 Poultry Meat Freshness Classification

An AI-powered web application for classifying poultry meat freshness using deep learning with ResNet transfer learning.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Overview

This project uses a ResNet-based deep learning model to classify poultry meat as either **Fresh (Segar)** or **Spoiled (Busuk)**. The application features a user-friendly Streamlit interface for real-time predictions.

## ✨ Features

- 🎯 **High Accuracy**: ~83% classification accuracy
- 🖼️ **Real-time Predictions**: Upload images and get instant results
- 📊 **Confidence Scores**: Visual confidence metrics with progress bars
- 🎨 **Beautiful UI**: Modern, responsive design with color-coded results
- 💡 **Safety Recommendations**: Actionable advice based on predictions
- 🔄 **Transfer Learning**: Leverages pre-trained ResNet50 architecture

## 🚀 Demo

![App Screenshot](images/demo.png)

## 📁 Project Structure

```
Poultry-Meat-Freshness-Classification/
├── app.py                          # Streamlit web application
├── train_new_model.py              # Model training script
├── test_actual_predictions.py      # Model testing script
├── models/
│   ├── model_trained_new.keras     # Trained model (Keras format)
│   └── model_trained_new.h5        # Trained model (H5 format)
├── notebooks/
│   └── Topsus_Poultry_Meat_ResNet.ipynb  # Training notebook
├── training_history/               # Training logs and metrics
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
└── README.md                       # Project documentation
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/Poultry-Meat-Freshness-Classification.git
cd Poultry-Meat-Freshness-Classification
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download the dataset** (if training from scratch)
- Place your dataset in `dataset_citra_dada_ayam/dataset 200x200/training/`
- Images should be named with prefixes: `busuk_*.jpg` for spoiled, `segar_*.jpg` for fresh

## 🎮 Usage

### Running the Web Application

```bash
streamlit run app.py
```

Or using Python module:
```bash
python -m streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Training a New Model

```bash
python train_new_model.py
```

### Testing Model Predictions

```bash
python test_actual_predictions.py
```

## 🧠 Model Architecture

- **Base Model**: ResNet50 (pre-trained on ImageNet)
- **Custom Layers**:
  - GlobalAveragePooling2D
  - Dropout (0.5)
  - Dense (128 units, ReLU activation)
  - Dropout (0.3)
  - Dense (1 unit, Sigmoid activation)

## 📊 Model Performance

- **Training Accuracy**: ~97%
- **Validation Accuracy**: ~83%
- **Dataset**: 1000 images (500 fresh + 500 spoiled)
- **Training Split**: 80% training, 20% validation
- **Epochs**: 10
- **Optimizer**: Adam (learning rate: 0.0001)

### Classification Threshold

- **Optimal Threshold**: 0.5445
- **Busuk (Spoiled)**: Prediction < 0.5445
- **Segar (Fresh)**: Prediction ≥ 0.5445

## 🖼️ How It Works

1. **Upload Image**: User uploads a poultry meat image
2. **Preprocessing**: Image is resized to 224x224 and normalized
3. **Prediction**: ResNet model processes the image
4. **Classification**: Output is compared against threshold
5. **Display Results**: Shows classification with confidence score

## 📦 Dependencies

- TensorFlow 2.x
- Streamlit
- NumPy
- Pillow
- h5py

See `requirements.txt` for complete list.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Aditi Raj** - *Initial work*

## 🙏 Acknowledgments

- ResNet architecture by Microsoft Research
- Dataset: Poultry Meat Freshness Dataset
- Built with Streamlit and TensorFlow
