
from pathlib import Path
import hashlib

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image


# Class names and labels
CLASSES = ["normal", "benign", "malignant"]


# Transformations for training images
train_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Transformations for validation and test images
eval_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Collect original images and their labels
def collect_samples(root):

    root = Path(root)
    samples = []

    for class_name in CLASSES:

        class_folder = root / class_name

        images = [
            path for path in class_folder.glob("*.png")
            if "_mask" not in path.name.lower()
        ]

        label = CLASSES.index(class_name)

        for path in images:
            samples.append((path, label))

    return samples


# Calculate SHA-256 hash of an image
def get_file_hash(path):

    sha256 = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


# Remove exact duplicates safely
def remove_duplicates(samples):

    hash_groups = {}

    for path, label in samples:

        image_hash = get_file_hash(path)

        if image_hash not in hash_groups:
            hash_groups[image_hash] = []

        hash_groups[image_hash].append((path, label))

    unique_samples = []
    conflicting_groups = 0
    conflicting_images = 0
    duplicate_count = 0

    for image_hash, group in hash_groups.items():

        labels = {label for _, label in group}

        # Exclude all copies if identical images have conflicting labels
        if len(labels) > 1:

            conflicting_groups += 1
            conflicting_images += len(group)

            print("\nConflicting group excluded:")

            for path, label in group:
                print(CLASSES[label], ":", path.name)

            continue

        # Keep one copy for same-label duplicates
        unique_samples.append(group[0])
        duplicate_count += len(group) - 1

    print("\nDuplicate detection summary")
    print("---------------------------")
    print("Original images:", len(samples))
    print("Unique usable images:", len(unique_samples))
    print("Same-label duplicates removed:", duplicate_count)
    print("Conflicting groups excluded:", conflicting_groups)
    print("Images excluded due to conflicting labels:", conflicting_images)

    return unique_samples


# Split data into training, validation, and testing
def split_dataset(samples, seed=42):

    labels = [label for _, label in samples]

    train_samples, temp_samples = train_test_split(
        samples,
        test_size=0.30,
        random_state=seed,
        stratify=labels
    )

    temp_labels = [label for _, label in temp_samples]

    val_samples, test_samples = train_test_split(
        temp_samples,
        test_size=0.50,
        random_state=seed,
        stratify=temp_labels
    )

    return train_samples, val_samples, test_samples


# PyTorch Dataset class
class BUSIDataset(Dataset):

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("L")

        if self.transform:
            image = self.transform(image)

        return image, label


# Create DataLoaders
def create_dataloaders(root, batch_size=16, seed=42):

    samples = collect_samples(root)

    print("Images collected:", len(samples))

    samples = remove_duplicates(samples)

    train_samples, val_samples, test_samples = split_dataset(
        samples,
        seed=seed
    )

    train_dataset = BUSIDataset(train_samples, train_tf)
    val_dataset = BUSIDataset(val_samples, eval_tf)
    test_dataset = BUSIDataset(test_samples, eval_tf)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    print("\nDataset split summary")
    print("---------------------")
    print("Training:", len(train_dataset))
    print("Validation:", len(val_dataset))
    print("Testing:", len(test_dataset))

    return train_loader, val_loader, test_loader

