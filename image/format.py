import os
from PIL import Image

# Path to your dataset
dataset_path = r"C:\Users\91824\Desktop\Delivery Detective\image_dataset"

# Loop through each category folder (damaged, intact)
for category in os.listdir(dataset_path):
    category_path = os.path.join(dataset_path, category)
    if not os.path.isdir(category_path):
        continue

    print(f"Processing category: {category}")

    for file_name in os.listdir(category_path):
        file_path = os.path.join(category_path, file_name)
        
        # Check if file is an image
        if not file_name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue

        try:
            # Open and convert to RGB
            img = Image.open(file_path).convert("RGB")

            # Define new name with .jpg extension
            new_file_name = os.path.splitext(file_name)[0] + ".jpg"
            new_file_path = os.path.join(category_path, new_file_name)

            # Save image as .jpg
            img.save(new_file_path, "JPEG")

            # If original file was not .jpg, remove it
            if not file_name.lower().endswith(".jpg"):
                os.remove(file_path)

        except Exception as e:
            print(f"❌ Error converting {file_name}: {e}")

print("✅ All images converted to JPG successfully!")
