import json

# Load metadata
with open('data/breakhis_training_metadata.json', 'r') as f:
    data = json.load(f)

print(f"📊 Dataset Statistics:")
print(f"Total images: {len(data)}")
print(f"Benign: {len([x for x in data if x['class']=='benign'])}")
print(f"Malignant: {len([x for x in data if x['class']=='malignant'])}")

# Calculate splits
total = len(data)
train_size = int(0.7 * total)
val_size = int(0.15 * total)
test_size = total - train_size - val_size

print(f"\n📈 Data Splits:")
print(f"Training: {train_size} images ({train_size/total*100:.1f}%)")
print(f"Validation: {val_size} images ({val_size/total*100:.1f}%)")
print(f"Testing: {test_size} images ({test_size/total*100:.1f}%)")

# Calculate batches with batch size 64
batch_size = 64
train_batches = (train_size + batch_size - 1) // batch_size  # Ceiling division

print(f"\n🚀 Training Details:")
print(f"Batch size: {batch_size}")
print(f"Training batches: {train_batches}")
print(f"Images per batch: {batch_size}")
print(f"Total training images processed: {train_batches * batch_size}")
print(f"Actual training images: {train_size}")
print(f"Last batch size: {train_size % batch_size if train_size % batch_size != 0 else batch_size}")
