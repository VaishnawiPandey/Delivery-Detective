import os
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Paths
model_path = r"C:\Users\91824\Desktop\Delivery Detective\cnn_package_model.h5"
new_images_path = r"C:\Users\91824\Desktop\Delivery Detective\new_images"
output_csv = r"C:\Users\91824\Desktop\Delivery Detective\package_predictions.csv"

# Image parameters
IMG_HEIGHT, IMG_WIDTH = 224, 224

# Load trained CNN model
model = load_model(model_path)
print("✅ Model loaded successfully!")

# Prepare list to store predictions
predictions = []

# Process each image in the new_images folder
for img_file in os.listdir(new_images_path):
    if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
        try:
            img_path = os.path.join(new_images_path, img_file)
            
            # Load and preprocess image
            img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
            img_array = img_to_array(img) / 255.0  # normalize
            img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

            # Predict
            pred_prob = model.predict(img_array)[0][0]
            label = "damaged" if pred_prob >= 0.15 else "intact"

            # Append result
            predictions.append({
                "image": img_file,
                "predicted_label": label,
                "probability_damaged": float(pred_prob)
            })

        except Exception as e:
            print(f"❌ Error processing {img_file}: {e}")

# Save predictions to CSV
df = pd.DataFrame(predictions)
df.to_csv(output_csv, index=False)
print(f"✅ Predictions saved to {output_csv}")
