from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import glob

print("Loading model...")
model = load_model('models/model_trained_new.keras', compile=False)

print("\nTesting ALL BUSUK images...")
busuk_files = glob.glob('dataset_citra_dada_ayam/dataset 200x200/training/busuk*.jpg')
busuk_preds = []
for f in busuk_files:
    img = load_img(f, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array, verbose=0)[0][0]
    busuk_preds.append(pred)

print(f"Tested {len(busuk_preds)} busuk images")
print(f"  Min: {min(busuk_preds):.4f}")
print(f"  Max: {max(busuk_preds):.4f}")
print(f"  Average: {np.mean(busuk_preds):.4f}")
print(f"  Median: {np.median(busuk_preds):.4f}")

print("\nTesting ALL SEGAR images...")
segar_files = glob.glob('dataset_citra_dada_ayam/dataset 200x200/training/segar*.jpg')
segar_preds = []
for f in segar_files:
    img = load_img(f, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array, verbose=0)[0][0]
    segar_preds.append(pred)

print(f"Tested {len(segar_preds)} segar images")
print(f"  Min: {min(segar_preds):.4f}")
print(f"  Max: {max(segar_preds):.4f}")
print(f"  Average: {np.mean(segar_preds):.4f}")
print(f"  Median: {np.median(segar_preds):.4f}")

# Calculate optimal threshold
optimal_threshold = (np.mean(busuk_preds) + np.mean(segar_preds)) / 2

print("\n" + "="*60)
print("ANALYSIS:")
print(f"Optimal threshold: {optimal_threshold:.4f}")
print(f"Busuk average: {np.mean(busuk_preds):.4f}")
print(f"Segar average: {np.mean(segar_preds):.4f}")

if np.mean(busuk_preds) < np.mean(segar_preds):
    print("\n✓ CORRECT: Lower values = BUSUK, Higher values = SEGAR")
    print(f"  Use: if pred < {optimal_threshold:.4f} → BUSUK, else → SEGAR")
else:
    print("\n✓ INVERTED: Higher values = BUSUK, Lower values = SEGAR")
    print(f"  Use: if pred >= {optimal_threshold:.4f} → BUSUK, else → SEGAR")

# Test accuracy
correct = 0
total = len(busuk_preds) + len(segar_preds)

for pred in busuk_preds:
    if (np.mean(busuk_preds) < np.mean(segar_preds) and pred < optimal_threshold) or \
       (np.mean(busuk_preds) >= np.mean(segar_preds) and pred >= optimal_threshold):
        correct += 1

for pred in segar_preds:
    if (np.mean(busuk_preds) < np.mean(segar_preds) and pred >= optimal_threshold) or \
       (np.mean(busuk_preds) >= np.mean(segar_preds) and pred < optimal_threshold):
        correct += 1

accuracy = (correct / total) * 100
print(f"\nPredicted accuracy with threshold {optimal_threshold:.4f}: {accuracy:.2f}%")
print("="*60)
