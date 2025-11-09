# 🍗 Poultry Meat Freshness Classifier - Streamlit App

A beautiful and interactive web application for classifying poultry meat freshness using deep learning.

## Features

- **User-Friendly Interface**: Clean, modern design with intuitive controls
- **Real-Time Classification**: Instant AI-powered freshness detection
- **Detailed Results**: Confidence scores and probability metrics
- **Visual Feedback**: Color-coded results (green for fresh, red for spoiled)
- **Responsive Design**: Works on desktop and mobile devices
- **Image Preview**: View uploaded images before analysis
- **Safety Recommendations**: Get actionable advice based on results

## Installation

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Verify Model File**
Make sure the model file exists at:
```
models/model_resnet_lr7e-4_m0_e200.h5
```

## Running the App

**Option 1: Using Python module**
```bash
python -m streamlit run app.py
```

**Option 2: Using the batch file (Windows)**
```bash
run_app.bat
```

The app will open in your default browser at `http://localhost:8501`

**Note:** If `streamlit` command is not recognized, always use `python -m streamlit` instead.

## How to Use

1. **Upload Image**: Click "Browse files" or drag and drop a poultry meat image
2. **Wait for Analysis**: The AI will process the image automatically
3. **View Results**: See the classification (Fresh/Spoiled) with confidence score
4. **Read Recommendations**: Follow the safety advice provided

## Model Information

- **Architecture**: ResNet with Transfer Learning
- **Input Size**: 224x224 pixels (RGB)
- **Classes**: 
  - Segar (Fresh)
  - Busuk (Spoiled)
- **Training**: 200 epochs with learning rate 7e-4

## Tips for Best Results

- Use clear, well-lit images
- Focus on the meat surface texture
- Avoid blurry or dark photos
- Ensure the meat is clearly visible
- Use recent photos for accurate assessment

## Technical Details

The app uses:
- **Streamlit**: For the web interface
- **TensorFlow/Keras**: For model inference
- **PIL**: For image processing
- **NumPy**: For numerical operations

## Disclaimer

⚠️ This is an AI-based educational tool. Always consult food safety experts and follow proper food handling guidelines. Do not rely solely on this tool for food safety decisions.

## Troubleshooting

**Model not loading?**
- Check if the model file path is correct
- Ensure TensorFlow is properly installed

**Image upload issues?**
- Supported formats: JPG, JPEG, PNG
- Try reducing image size if too large

**Slow predictions?**
- First prediction may be slower (model loading)
- Subsequent predictions will be faster

## License

See LICENSE file in the main repository.
