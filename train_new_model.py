"""
Train a new ResNet model from scratch with your dataset
"""
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import numpy as np

print("="*60)
print("Training New ResNet Model")
print("="*60)

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# Load images from directory
# Since images are in one folder with filename prefixes, we need to organize them
import os
import shutil
import glob

# Create temporary directory structure
train_dir = 'temp_training'
if os.path.exists(train_dir):
    shutil.rmtree(train_dir)

os.makedirs(f'{train_dir}/busuk', exist_ok=True)
os.makedirs(f'{train_dir}/segar', exist_ok=True)

print("\nOrganizing dataset...")
# Copy busuk images
busuk_files = glob.glob('dataset_citra_dada_ayam/dataset 200x200/training/busuk*.jpg')
for i, f in enumerate(busuk_files):
    shutil.copy(f, f'{train_dir}/busuk/busuk_{i}.jpg')

# Copy segar images  
segar_files = glob.glob('dataset_citra_dada_ayam/dataset 200x200/training/segar*.jpg')
for i, f in enumerate(segar_files):
    shutil.copy(f, f'{train_dir}/segar/segar_{i}.jpg')

print(f"✓ Organized {len(busuk_files)} busuk images")
print(f"✓ Organized {len(segar_files)} segar images")

# Create generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

print(f"\nClass indices: {train_generator.class_indices}")
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {validation_generator.samples}")

# Build model
print("\nBuilding model...")
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base model
for layer in base_model.layers:
    layer.trainable = False

# Add custom head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("✓ Model built and compiled")

# Train
print("\nTraining model...")
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    verbose=1
)

# Save model
print("\nSaving model...")
model.save('models/model_trained_new.h5')
model.save('models/model_trained_new.keras')
print("✓ Model saved!")

# Test predictions
print("\nTesting predictions...")
test_busuk = glob.glob(f'{train_dir}/busuk/*.jpg')[0]
test_segar = glob.glob(f'{train_dir}/segar/*.jpg')[0]

from tensorflow.keras.preprocessing.image import load_img, img_to_array

for test_file, expected in [(test_busuk, 'BUSUK'), (test_segar, 'SEGAR')]:
    img = load_img(test_file, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array, verbose=0)[0][0]
    
    # Based on class_indices
    if train_generator.class_indices['busuk'] == 0:
        result = "BUSUK" if pred < 0.5 else "SEGAR"
    else:
        result = "SEGAR" if pred < 0.5 else "BUSUK"
    
    status = "✓" if result == expected else "✗"
    print(f"{status} {os.path.basename(test_file)}: {pred:.4f} → {result}")

print("\n" + "="*60)
print("Training complete!")
print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
print("="*60)

# Cleanup
shutil.rmtree(train_dir)
print("\n✓ Cleaned up temporary files")
