import os
import random
from PIL import Image, ImageEnhance
import shutil

# Paths
raw_dataset_path = r"C:\Users\91824\Desktop\Delivery Detective\image_dataset"
processed_dataset_path = r"C:\Users\91824\Desktop\Delivery Detective\processed_dataset"

# Image size
IMG_SIZE = (224, 224)

# Train / Val / Test split
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Augmentation function
def augment_image(img):
    # Random horizontal flip
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # Random rotation
    angle = random.uniform(-15, 15)
    img = img.rotate(angle)
    # Random brightness
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.8, 1.2))
    return img

# Create processed dataset folders
for split in ['train', 'val', 'test']:
    for category in ['damaged', 'intact']:
        os.makedirs(os.path.join(processed_dataset_path, split, category), exist_ok=True)

# Process images
for category in ['damaged', 'intact']:
    category_path = os.path.join(raw_dataset_path, category)
    images = [f for f in os.listdir(category_path) if f.lower().endswith(".jpg")]
    random.shuffle(images)

    total = len(images)
    train_end = int(total * TRAIN_SPLIT)
    val_end = train_end + int(total * VAL_SPLIT)

    for i, img_name in enumerate(images):
        img_path = os.path.join(category_path, img_name)
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize(IMG_SIZE)

            # Normalize by scaling pixel values (0-255 → 0-1)
            img = img  # actual normalization can be done in model pipeline if using tensors

            # Apply augmentation only for training set
            if i < train_end:
                img_aug = augment_image(img)
                save_path = os.path.join(processed_dataset_path, 'train', category, img_name)
                img_aug.save(save_path)
            elif i < val_end:
                save_path = os.path.join(processed_dataset_path, 'val', category, img_name)
                img.save(save_path)
            else:
                save_path = os.path.join(processed_dataset_path, 'test', category, img_name)
                img.save(save_path)

        except Exception as e:
            print(f"❌ Error processing {img_name}: {e}")

print("✅ Preprocessing complete! Dataset ready at 'processed_dataset/'")
