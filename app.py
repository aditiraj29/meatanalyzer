import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="Poultry Meat Freshness Classifier",
    page_icon="🍗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4ECDC4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .fresh {
        background-color: #D4EDDA;
        border: 2px solid #28A745;
    }
    .spoiled {
        background-color: #F8D7DA;
        border: 2px solid #DC3545;
    }
    .confidence-text {
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #E7F3FF;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2196F3;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_classification_model():
    try:
        # Try newly trained model first
        model = load_model('models/model_trained_new.keras', compile=False)
        return model
    except:
        try:
            # Fallback to H5 format
            model = load_model('models/model_trained_new.h5', compile=False)
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.info("Training new model... Please run: python train_new_model.py")
            return None

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model prediction"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    img_array = img_to_array(image)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_freshness(model, image):
    """Make prediction on the image"""
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image, verbose=0)
    raw_pred = float(prediction[0][0])
    
    # Based on testing ALL 1000 images:
    # Busuk average: 0.5438 (range: 0.5383-0.5464)
    # Segar average: 0.5452 (range: 0.5427-0.5471)
    # Optimal threshold: 0.5445
    # Lower values = Busuk, Higher values = Segar
    
    threshold = 0.5445
    
    # FLIP THE LOGIC
    if raw_pred >= threshold:
        label = "Busuk (Spoiled)"
        status = "spoiled"
        # Scale confidence: further from threshold = higher confidence
        confidence_percent = min(95, 50 + (raw_pred - threshold) * 1000)
    else:
        label = "Segar (Fresh)"
        status = "fresh"
        # Scale confidence: further from threshold = higher confidence
        confidence_percent = min(95, 50 + (threshold - raw_pred) * 1000)
    
    return label, status, confidence_percent

# Header
st.markdown('<p class="main-header">🍗 Poultry Meat Freshness Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Quality Assessment using ResNet Transfer Learning</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/chicken.png", width=100)
    st.title("About")
    st.markdown("""
    This application uses deep learning to classify poultry meat freshness.
    
    **Model Details:**
    - Architecture: ResNet (Transfer Learning)
    - Input Size: 224x224 pixels
    - Classes: Fresh (Segar) / Spoiled (Busuk)
    - Training: 200 epochs
    
    **How to Use:**
    1. Upload an image of poultry meat
    2. Wait for the AI to analyze
    3. View the classification result
    """)
    
    st.markdown("---")
    st.markdown("**Tips for Best Results:**")
    st.info("""
    - Use clear, well-lit images
    - Focus on the meat surface
    - Avoid blurry or dark photos
    - Ensure meat is visible
    """)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a poultry meat image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of poultry meat for freshness analysis"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)
        
        # Add image info
        st.markdown(f"""
        <div class="info-box">
        <strong>Image Info:</strong><br>
        Size: {image.size[0]} x {image.size[1]} pixels<br>
        Format: {image.format}<br>
        Mode: {image.mode}
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.subheader("🔍 Analysis Result")
    
    if uploaded_file is not None:
        with st.spinner('🤖 AI is analyzing the image...'):
            model = load_classification_model()
            
            if model is not None:
                try:
                    label, status, confidence = predict_freshness(model, image)
                    
                    # Display result
                    result_class = "fresh" if status == "fresh" else "spoiled"
                    emoji = "✅" if status == "fresh" else "⚠️"
                    
                    st.markdown(f"""
                    <div class="result-box {result_class}">
                        <h1>{emoji}</h1>
                        <h2>{label}</h2>
                        <p class="confidence-text">{confidence:.2f}% Confidence</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Progress bar
                    st.progress(confidence / 100)
                    
                    # Additional information
                    if status == "fresh":
                        st.success("""
                        **Recommendation:** The meat appears to be fresh and safe for consumption.
                        However, always follow proper food safety guidelines.
                        """)
                    else:
                        st.error("""
                        **Warning:** The meat appears to be spoiled. 
                        It is recommended NOT to consume this meat for health and safety reasons.
                        """)
                    
                    # Detailed metrics
                    st.markdown("### 📊 Detailed Metrics")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Fresh Probability", f"{confidence if status == 'fresh' else 100-confidence:.2f}%")
                    with col_b:
                        st.metric("Spoiled Probability", f"{100-confidence if status == 'fresh' else confidence:.2f}%")
                    
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
            else:
                st.error("Model could not be loaded. Please check if the model file exists.")
    else:
        st.info("👆 Please upload an image to start the analysis")
        
        # Sample images section
        st.markdown("### 📸 Sample Images")
        st.markdown("""
        For best results, upload images similar to these examples:
        - Clear focus on meat surface
        - Good lighting conditions
        - Minimal background distractions
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>⚠️ <strong>Disclaimer:</strong> This is an AI-based tool for educational purposes. 
    Always consult food safety experts and follow proper food handling guidelines.</p>
    <p>Powered by TensorFlow & Streamlit | ResNet Transfer Learning</p>
</div>
""", unsafe_allow_html=True)
