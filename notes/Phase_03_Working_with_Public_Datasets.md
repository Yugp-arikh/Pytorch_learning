# Phase 3: Working with Public Datasets — The PyTorch Data Ecosystem

> **Covers**: Chapter 4 of *AI and ML for Coders in PyTorch* by Laurence Moroney
> **Prerequisites**: Phase 1-2 (tensors, training loop, DataLoader basics, transforms)
> **Goal**: Master the complete data pipeline in PyTorch — from raw data to model-ready batches. Understand Dataset, DataLoader, transforms, splits, the ETL pattern, and hardware-aware performance optimization.

---

## Table of Contents

1. [Why Data Management Matters](#1-why-data-management-matters)
2. [The PyTorch Data Ecosystem — Domain Libraries](#2-the-pytorch-data-ecosystem--domain-libraries)
3. [torch.utils.data.Dataset — The Foundation Class](#3-torchutilsdatadataset--the-foundation-class)
4. [Building a Custom Dataset from Scratch](#4-building-a-custom-dataset-from-scratch)
5. [Exploring the FashionMNIST Dataset Class](#5-exploring-the-fashionmnist-dataset-class)
6. [Generic Dataset Classes — ImageFolder, DatasetFolder, FakeData](#6-generic-dataset-classes--imagefolder-datasetfolder-fakedata)
7. [Custom Splits with random_split](#7-custom-splits-with-random_split)
8. [The ETL Pattern — Extract, Transform, Load](#8-the-etl-pattern--extract-transform-load)
9. [The DataLoader Class — Deep Dive](#9-the-dataloader-class--deep-dive)
10. [Hardware-Aware Pipelining — CPU vs GPU/TPU](#10-hardware-aware-pipelining--cpu-vs-gputpu)
11. [Parallelizing Data Loading with num_workers](#11-parallelizing-data-loading-with-num_workers)
12. [Complete Example: CIFAR-10 with Parallel ETL](#12-complete-example-cifar-10-with-parallel-etl)
13. [Glossary](#13-glossary)
14. [Summary](#14-summary)
15. [Review Questions](#15-review-questions)

---

## 1. Why Data Management Matters

In Phases 1-2, we focused on model architectures and training loops. But in real-world ML, **the data pipeline often requires more code and engineering effort than the model itself**. Consider what we've already encountered:

| Chapter | Data Source | How We Got It |
|---------|-----------|---------------|
| Ch 1 | 6 hand-typed (x, y) pairs | Hardcoded into Python |
| Ch 2 | Fashion MNIST (70K images) | `datasets.FashionMNIST(download=True)` |
| Ch 3 | Horses/Humans (1K images) | Downloaded ZIP, extracted to directories |
| Ch 3 | Dogs vs Cats (25K images) | Downloaded from Kaggle, extracted to directories |
| Ch 3 | Rock/Paper/Scissors | Downloaded ZIP from Google Storage |

Each dataset required a different method of acquisition, preprocessing, and loading. PyTorch's data ecosystem provides a **unified API** that handles all of this consistently — whether your dataset fits in RAM or spans terabytes across distributed storage.

---

## 2. The PyTorch Data Ecosystem — Domain Libraries

PyTorch organizes datasets and tools into **domain-specific libraries**, each tailored to a particular type of data:

### 2.1 torchvision (Computer Vision)

```python
from torchvision import datasets, transforms, models
```

| Component | What It Provides |
|-----------|-----------------|
| `datasets` | Image Classification (MNIST, CIFAR, ImageNet...), Object Detection (COCO, VOC...), Segmentation, Optical Flow, Stereo Matching, Image Captioning, Video Classification |
| `transforms` | Resize, Normalize, RandomCrop, RandomFlip, ColorJitter, ToTensor, etc. |
| `models` | Pretrained architectures (ResNet, Inception, VGG, MobileNet, etc.) |

### 2.2 torchtext (Natural Language Processing)

```python
from torchtext import datasets, vocab
```

| Component | What It Provides |
|-----------|-----------------|
| `datasets` | Text Classification, Language Modeling, Machine Translation, Sequence Tagging, Question Answering, Unsupervised Learning |
| `vocab` | Vocabulary building, pretrained word vectors (GloVe, FastText) |

> **Note**: `torchtext` has been deprecated since 2023. Modern NLP workflows typically use Hugging Face `transformers` and `datasets` libraries instead. The book acknowledges this and uses Hugging Face tokenizers in later chapters.

### 2.3 torchaudio (Audio/Speech)

```python
from torchaudio import datasets, transforms
```

| Component | What It Provides |
|-----------|-----------------|
| `datasets` | Speech recognition datasets (LibriSpeech, CommonVoice, etc.) |
| `transforms` | Spectrogram, MelSpectrogram, MFCC, Resample, etc. |

### 2.4 The Common Thread

**All datasets in every domain library are subclasses of `torch.utils.data.Dataset`.** This means they all share the same interface and all work with `DataLoader`. Learn the base class, and you can use any dataset in any domain.

---

## 3. torch.utils.data.Dataset — The Foundation Class

This is the abstract base class that every PyTorch dataset must implement. It lives in:

```python
from torch.utils.data import Dataset
```

### 3.1 The Contract

To create a valid dataset, you must implement exactly **two methods**:

| Method | Signature | Returns | Purpose |
|--------|----------|---------|---------|
| `__len__` | `def __len__(self) -> int` | Integer | Total number of samples in the dataset |
| `__getitem__` | `def __getitem__(self, index) -> Any` | One sample | The sample at position `index` |

That's it. These two methods are the only requirement. Everything else — batching, shuffling, parallel loading, etc. — is handled by `DataLoader`.

### 3.2 How Python Uses These Methods

These are **Python dunder (double-underscore) methods** that integrate with Python's built-in syntax:

```python
dataset = MyDataset(...)

len(dataset)      # Calls dataset.__len__()
dataset[42]       # Calls dataset.__getitem__(42)
```

`DataLoader` uses `__len__` to know how many batches to create and `__getitem__` to fetch individual samples.

### 3.3 Minimal Example

```python
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, transforms=None):
        self.data = data
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transforms:
            sample = self.transforms(sample)
        return sample
```

**Key design decisions in `__init__`**:
- Store the raw data (or paths to data on disk)
- Store the transform pipeline
- Do NOT load all data into memory here if the dataset is large — load lazily in `__getitem__`

**Key design decision in `__getitem__`**:
- Fetch one sample by index
- Apply transforms on-the-fly (not pre-applied)
- Return the transformed sample (and usually its label)

---

## 4. Building a Custom Dataset from Scratch

### 4.1 Example: A Simple Linear Dataset

Suppose we want to recreate the `y = 2x - 1` problem from Chapter 1, but as a proper dataset:

```python
import torch
from torch.utils.data import Dataset, DataLoader

# Generate synthetic data
torch.manual_seed(0)   # For reproducibility
x = torch.arange(0, 100, dtype=torch.float32)
y = 2 * x - 1
```

### 4.2 The Dataset Class

```python
class LinearDataset(Dataset):
    def __init__(self, x, y):
        """
        Initialize the dataset with x (inputs) and y (labels).
        Args:
            x (torch.Tensor): Input features, shape (N,)
            y (torch.Tensor): Output labels, shape (N,)
        """
        self.x = x
        self.y = y

    def __len__(self):
        """Return the total number of samples."""
        return len(self.x)

    def __getitem__(self, idx):
        """
        Fetch the sample at index `idx`.
        Args:
            idx (int): Index of the sample
        Returns:
            tuple: (input_value, label_value)
        """
        return self.x[idx], self.y[idx]
```

### 4.3 Using the Dataset with DataLoader

```python
# Create an instance
dataset = LinearDataset(x, y)

# Wrap it in a DataLoader
data_loader = DataLoader(dataset, batch_size=10, shuffle=True)

# Iterate
for batch_idx, (inputs, labels) in enumerate(data_loader):
    print(f"Batch {batch_idx + 1}")
    print("  Inputs:", inputs)
    print("  Labels:", labels)
    if batch_idx == 0:
        break
```

**Output** (example — order varies due to `shuffle=True`):
```
Batch 1
  Inputs: tensor([47., 83., 12., 91., 35., 6., 72., 54., 28., 61.])
  Labels: tensor([ 93., 165.,  23., 181.,  69.,  11., 143., 107.,  55., 121.])
```

### 4.4 The Pattern

Every custom dataset follows the same structure:

```
class MyDataset(Dataset):
    __init__   → Store data references and transform pipelines
    __len__    → Return count of samples
    __getitem__ → Return one (input, label) pair, transformed
```

This pattern scales from 6 data points to billions of records. For large datasets, `__init__` might store file paths instead of data, and `__getitem__` loads from disk on demand.

---

## 5. Exploring the FashionMNIST Dataset Class

Fashion MNIST is a pre-built dataset class in `torchvision.datasets`. Let's understand what happens when you call it:

```python
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)
```

### 5.1 What Each Parameter Does

| Parameter | Value | Effect |
|-----------|-------|--------|
| `root='./data'` | Directory path | Where to store/find the downloaded files |
| `train=True` | Boolean | `True` → 60,000 training images; `False` → 10,000 test images |
| `download=True` | Boolean | Download from the internet if not found locally |
| `transform=transform` | Transform pipeline | Applied to each image when `__getitem__` is called |

### 5.2 What Happens Internally

1. **Download**: If `download=True` and files aren't at `root`, downloads them from the Fashion MNIST server
2. **Store**: Saves the raw data as binary files on disk
3. **Index**: Creates an internal index mapping each integer index to its (image, label) pair
4. **On access**: When you call `dataset[i]`, it:
   - Loads the raw image bytes for index `i`
   - Converts to a PIL Image
   - Applies the `transform` pipeline (e.g., `ToTensor()` normalizes to [0,1])
   - Returns `(transformed_image, label)`

### 5.3 Inspecting the Dataset

```python
print(len(train_dataset))           # 60000
print(train_dataset[0][0].shape)    # torch.Size([1, 28, 28])  ← image tensor
print(train_dataset[0][1])          # 9  ← label (Ankle Boot)
print(train_dataset.classes)        # ['T-shirt/top', 'Trouser', ..., 'Ankle boot']
```

---

## 6. Generic Dataset Classes — ImageFolder, DatasetFolder, FakeData

When your data doesn't come from a built-in dataset class, PyTorch provides generic classes that handle common patterns.

### 6.1 ImageFolder

**Use case**: Images organized in subdirectories, where each subdirectory name is a class label.

**Directory structure**:
```
training/
├── horses/          ← class 0 (alphabetical)
│   ├── horse001.png
│   ├── horse002.png
│   └── ...
└── humans/          ← class 1 (alphabetical)
    ├── human001.png
    ├── human002.png
    └── ...
```

**Code**:
```python
from torchvision.datasets import ImageFolder

train_dataset = ImageFolder(root='training/', transform=transform)
```

**How it works**:
1. Scans all subdirectories under `root`
2. Sorts subdirectory names **alphabetically** → these become class labels
3. Assigns integer indices: first alphabetically = 0, second = 1, etc.
4. Each image file becomes one sample

**Critical gotcha — alphabetical ordering**:

| Subdirectory | Expected Class | Actual Class (alphabetical) |
|-------------|---------------|---------------------------|
| Rock | 0 | 1 |
| Paper | 1 | 0 |
| Scissors | 2 | 2 |

"Rock, Paper, Scissors" in common parlance suggests (0, 1, 2), but alphabetically it's **Paper=0, Rock=1, Scissors=2**. This can cause confusion when interpreting outputs.

**Fix — custom class-to-index mapping**:
```python
custom_class_to_idx = {'rock': 0, 'paper': 1, 'scissors': 2}
dataset = ImageFolder(
    root='data/rps',
    target_transform=lambda x: custom_class_to_idx[dataset.classes[x]]
)
dataset.class_to_idx = custom_class_to_idx
```

### 6.2 DatasetFolder

**ImageFolder** is actually a subclass of the more general **DatasetFolder**, which works with any file type — not just images.

**Example**: Text files organized by class:
```
root/
├── sarcasm/
│   ├── document1.txt
│   └── document2.txt
└── factual/
    ├── factdoc1.txt
    └── factdoc2.txt
```

You provide a custom loader function that reads files, and `DatasetFolder` handles the directory-to-label mapping and integration with `DataLoader`.

### 6.3 FakeData

**Use case**: Quick prototyping and benchmarking when you don't have real data.

```python
from torchvision.datasets import FakeData

fake_dataset = FakeData(
    size=100,                     # 100 fake images
    image_size=(3, 224, 224),     # 3-channel, 224x224 (e.g., for MobileNet)
    num_classes=10,               # 10 random labels
    transform=transform
)

data_loader = DataLoader(fake_dataset, batch_size=10, shuffle=True)
```

**What it generates**: Random noise images with random labels. Useful for:
- Testing that your model architecture accepts the correct input shapes
- Benchmarking training speed
- Debugging data pipeline issues without downloading real data

> **Note**: FakeData only supports image data. For other data types (numeric, sequence), create a custom `Dataset` subclass with random data.

---

## 7. Custom Splits with random_split

### 7.1 The Problem

Some datasets come pre-split (Fashion MNIST: 60K train / 10K test). But what if you need:
- A three-way split (train / validation / test)?
- A different ratio (e.g., 80/10/10 instead of 85.7/14.3)?
- Multiple different splits for cross-validation?

### 7.2 The Solution: random_split

```python
from torch.utils.data import random_split

# Load the FULL dataset (no train/test split)
dataset = datasets.FashionMNIST(root='./data', download=True, transform=transform)
# Note: no `train=True` parameter — we get all 70,000 images

total_count = len(dataset)                              # 70,000
train_count = int(0.7 * total_count)                    # 49,000
val_count = int(0.15 * total_count)                     # 10,500
test_count = total_count - train_count - val_count      # 10,500

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_count, val_count, test_count]
)
```

### 7.3 How random_split Works

1. Takes the full dataset and a list of sizes `[n1, n2, n3]`
2. The sizes **must** sum to `len(dataset)` (otherwise error)
3. Randomly assigns each sample to one of the subsets
4. Returns `Subset` objects that behave like regular datasets
5. Each call with a different random seed produces a different split

### 7.4 Why the Third Split is Calculated as a Remainder

```python
test_count = total_count - train_count - val_count
```

When you compute `int(0.7 * 70000) = 49000` and `int(0.15 * 70000) = 10500`, the sum is `49000 + 10500 + 10500 = 70000`. But for other dataset sizes, floating-point arithmetic can leave remainders. Computing the last split as a subtraction guarantees **all data is used and none is wasted**.

### 7.5 Important Distinction

`random_split` **slices the dataset at random indices** — it does not shuffle the underlying data. Each subset still returns samples in their original order within that subset. If you want shuffled access, use `shuffle=True` in the `DataLoader`.

### 7.6 Why Multiple Splits Help

| Problem | How Multiple Splits Help |
|---------|------------------------|
| Overfitting detection | If accuracy is high on one split but low on another, the model may be memorizing |
| Architecture validation | Consistent results across different splits = robust architecture |
| Cross-validation | Train on K different splits and average results for more reliable metrics |

---

## 8. The ETL Pattern — Extract, Transform, Load

### 8.1 What is ETL?

**ETL (Extract, Transform, Load)** is a data engineering pattern that organizes data processing into three distinct phases. It's used throughout industry — from data warehouses to ML pipelines.

### 8.2 The Three Phases in ML

```
┌────────────────────┐    ┌────────────────────┐    ┌────────────────────┐
│     EXTRACT        │ →  │    TRANSFORM       │ →  │      LOAD          │
│                    │    │                    │    │                    │
│ • Download data    │    │ • Resize images    │    │ • Batch samples    │
│ • Read from disk   │    │ • Normalize pixels │    │ • Feed to model    │
│ • Decompress ZIP   │    │ • Augment images   │    │ • Execute on GPU   │
│ • Parse labels     │    │ • Tokenize text    │    │                    │
│                    │    │ • Convert to tensor│    │                    │
│   (Dataset class)  │    │   (transforms)     │    │   (DataLoader)     │
└────────────────────┘    └────────────────────┘    └────────────────────┘
         ↑                          ↑                         ↑
     Runs on CPU              Runs on CPU              Runs on GPU/TPU
```

### 8.3 ETL Mapped to PyTorch Code

Here's the "Horses or Humans" pipeline mapped to ETL:

```python
# ═══ TRANSFORM (defined first, applied during Extract) ═══
train_transform = transforms.Compose([
    transforms.Resize((150, 150)),         # T: Resize
    transforms.RandomHorizontalFlip(),     # T: Augment
    transforms.RandomRotation(20),         # T: Augment
    transforms.RandomAffine(               # T: Augment
        degrees=0,
        translate=(0.2, 0.2),
        scale=(0.8, 1.2),
        shear=20,
    ),
    transforms.ToTensor(),                 # T: Convert to tensor
    transforms.Normalize(                  # T: Normalize
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    ),
])

# ═══ EXTRACT (reads data from disk, applies transform) ═══
train_dataset = datasets.ImageFolder(
    root=training_dir,                     # E: Where to find data
    transform=train_transform              # T: Applied per sample
)

# ═══ LOAD (batches and feeds data to model) ═══
train_loader = DataLoader(
    train_dataset,                         # L: Data source
    batch_size=32,                         # L: How many per batch
    shuffle=True                           # L: Randomize order
)
```

### 8.4 Why ETL Matters for ML

**Consistency**: The same ETL pattern works whether your data is:
- 6 numbers in memory
- 70,000 images downloaded from a server
- Millions of images on distributed storage

**Separation of concerns**: Each phase has a clear responsibility:
- Extract: Get the data (disk I/O, network, decompression)
- Transform: Make the data model-ready (normalize, augment, convert)
- Load: Feed the data efficiently (batch, shuffle, parallelize)

**Scalability**: The pattern scales without changing your training code. You can swap `DataLoader(num_workers=1)` for `num_workers=8` without touching the model or training loop.

---

## 9. The DataLoader Class — Deep Dive

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,                # Any Dataset subclass
    batch_size=64,          # Samples per batch
    shuffle=True,           # Randomize order each epoch
    num_workers=4,          # Parallel data loading processes
    pin_memory=True,        # Speed up CPU → GPU transfer
    drop_last=False         # Whether to drop the last incomplete batch
)
```

### 9.1 Batching

**What it does**: Groups individual samples into batches.

**Why batch?**
1. **Memory efficiency**: GPUs have fixed memory. Batching ensures we use it optimally.
2. **Better gradients**: SGD computes gradients on a batch, giving a more stable estimate than a single sample.
3. **Parallelism**: GPU cores process all samples in a batch simultaneously.

**Batch arithmetic**:
```
dataset_size = 60,000
batch_size   = 64
num_batches  = ceil(60,000 / 64) = 938

Batch 1-937:  64 samples each  (937 × 64 = 59,968)
Batch 938:    32 samples       (60,000 - 59,968 = 32)
```

If `drop_last=True`, batch 938 (the incomplete one) is discarded. This ensures every batch has exactly `batch_size` samples, which can be important for some operations like BatchNorm.

### 9.2 Shuffling

**What it does**: Randomizes the order in which samples are drawn from the dataset at the start of each epoch.

**Why shuffle?**

Without shuffling, consider Fashion MNIST where the first 6,000 images are all T-shirts, the next 6,000 are all Trousers, etc. With `batch_size=1000`:
- Batch 1: All T-shirts → model only sees T-shirts
- Batch 2: All T-shirts → still only T-shirts
- ...
- Batch 7: All Trousers → model suddenly sees only Trousers

The model "forgets" T-shirts while learning Trousers. This is catastrophic for learning.

With shuffling:
- Batch 1: Mixed classes → model learns general patterns
- Batch 2: Mixed classes → reinforces learning

**Rule**: Always `shuffle=True` for training, `shuffle=False` for validation/test (reproducibility).

### 9.3 Parallel Data Loading

**What it does**: Spawns multiple worker subprocesses that load and preprocess data in parallel with training.

```python
DataLoader(dataset, num_workers=4)
```

Creates 4 worker processes. While the GPU trains on batch N, the workers pre-fetch and preprocess batches N+1 through N+4. When the GPU finishes batch N, the next batch is already waiting.

### 9.4 Custom Data Sampling

**What it does**: Controls the order in which samples are drawn from the dataset.

**Base class**: `torch.utils.data.Sampler`

Built-in samplers:
- `SequentialSampler`: Samples in order (0, 1, 2, ...)
- `RandomSampler`: Samples randomly
- `WeightedRandomSampler`: Over/under-samples classes (useful for imbalanced datasets)

Custom samplers are beyond the scope of this chapter but follow the same `__iter__` and `__len__` pattern as datasets.

### 9.5 pin_memory

```python
DataLoader(dataset, pin_memory=True)
```

**What it does**: Allocates data in "pinned" (page-locked) CPU memory, which enables faster CPU → GPU transfer via DMA (Direct Memory Access).

**When to use**: When training on GPU. Has no effect on CPU-only training.

### 9.6 DataLoader Output Shape

For an image dataset with `batch_size=64`:

```python
for images, labels in train_loader:
    print(images.shape)    # torch.Size([64, 1, 28, 28])  ← batch of images
    print(labels.shape)    # torch.Size([64])              ← batch of labels
    break
```

The DataLoader automatically **collates** (stacks) individual samples into batch tensors:
- 64 images of shape `(1, 28, 28)` → one tensor of shape `(64, 1, 28, 28)`
- 64 integer labels → one tensor of shape `(64,)`

---

## 10. Hardware-Aware Pipelining — CPU vs GPU/TPU

### 10.1 The Bottleneck Problem

Training involves two types of work:
1. **Data preparation** (Extract + Transform): CPU-bound — downloading, decompressing, resizing, normalizing
2. **Model training** (Load + Forward + Backward): GPU-bound — matrix multiplications, gradient computation

### 10.2 Naive Approach — Sequential

```
Time →
CPU:  [Prepare Batch 1]                [Prepare Batch 2]                [Prepare Batch 3]
GPU:                    [Train Batch 1]                  [Train Batch 2]
                        ↑ IDLE ↑                         ↑ IDLE ↑
```

The GPU sits idle while the CPU prepares each batch. The CPU sits idle while the GPU trains. Total time = sum of all preparation + all training.

### 10.3 Pipelined Approach — Parallel

```
Time →
CPU:  [Prepare Batch 1][Prepare Batch 2][Prepare Batch 3][Prepare Batch 4]
GPU:                    [Train Batch 1 ][Train Batch 2 ][Train Batch 3 ]
```

While the GPU trains batch 1, the CPU prepares batch 2. When training finishes, the next batch is already ready. Total time ≈ max(total_preparation, total_training) — much faster.

### 10.4 How PyTorch Implements This

The `DataLoader` with `num_workers > 0` creates worker processes that:
1. Run on CPU
2. Pre-fetch and preprocess upcoming batches in a queue
3. Hand off ready batches to the training loop (which runs on GPU)

**Moving data to GPU**:
```python
for images, labels in train_loader:
    images = images.to(device)    # CPU → GPU transfer
    labels = labels.to(device)
    
    outputs = model(images)       # Runs on GPU
    loss = criterion(outputs, labels)
    ...
```

The `.to(device)` call moves tensors from CPU memory to GPU memory. With `pin_memory=True`, this transfer is faster because pinned memory avoids an intermediate copy.

### 10.5 Choosing the Right Batch Size

| Batch Size | GPU Utilization | Gradient Quality | Memory Usage |
|-----------|----------------|-----------------|-------------|
| Too small (1-4) | Low — GPU cores are underused | Noisy gradients | Low |
| Medium (32-128) | Good — sweet spot for most tasks | Stable gradients | Moderate |
| Too large (1024+) | High — but may hit memory limits | Very stable but may not generalize as well | High |

**General guidance**:
- Start with 32 or 64
- Increase if GPU memory is underused
- Decrease if you get CUDA out-of-memory errors
- Powers of 2 (32, 64, 128, 256) are slightly more efficient due to hardware alignment

---

## 11. Parallelizing Data Loading with num_workers

### 11.1 How It Works

```python
DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
```

- `num_workers=0` (default): Data is loaded in the main process. No parallelism.
- `num_workers=4`: 4 worker subprocesses are spawned. Each loads batches independently.

### 11.2 How Many Workers?

| Guideline | Rationale |
|-----------|-----------|
| Start with `num_workers=2` | Safe for most systems |
| Try `num_workers = num_cpu_cores` | One worker per CPU core |
| Monitor GPU utilization | If GPU is idle between batches, increase workers |
| Monitor CPU/RAM | If RAM or CPU maxes out, decrease workers |

**On Windows**: `num_workers > 0` requires your training code to be inside an `if __name__ == '__main__':` guard due to how Python multiprocessing works on Windows.

### 11.3 What Workers Do

```
Main Process               Worker 1            Worker 2            Worker 3            Worker 4
─────────────              ─────────            ─────────            ─────────            ─────────
Request batch 1 ─────────→ Load batch 1
Request batch 2 ────────────────────────────→ Load batch 2
Request batch 3 ───────────────────────────────────────────────→ Load batch 3
Request batch 4 ──────────────────────────────────────────────────────────────────→ Load batch 4
                           ↓
Train batch 1 ◄──────────── Done!
                                                ↓
Train batch 2 ◄──────────────────────────────── Done!
  ...
```

Workers load batches concurrently while the main process trains.

---

## 12. Complete Example: CIFAR-10 with Parallel ETL

CIFAR-10 is another classic dataset: 60,000 32x32 color images in 10 classes.

### 12.1 Extract + Transform

```python
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),                              # [0,255] → [0,1]
    transforms.Normalize((0.5, 0.5, 0.5),              # Center: [0,1] → [-1,1]
                         (0.5, 0.5, 0.5))
])

# Load CIFAR10 dataset
dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
```

### 12.2 Load (with Parallel Workers)

```python
from torch.utils.data import DataLoader

data_loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4     # 4 parallel workers
)
```

### 12.3 Model + Training Loop

```python
import torch

# Simple DNN for CIFAR10 (not a CNN — just for demonstration)
model = torch.nn.Sequential(
    torch.nn.Linear(3 * 32 * 32, 500),   # 3072 inputs (3 channels × 32 × 32)
    torch.nn.ReLU(),
    torch.nn.Linear(500, 10)              # 10 output classes
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

def train(model, data_loader):
    model.train()
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        # Flatten: (batch, 3, 32, 32) → (batch, 3072)
        inputs = inputs.view(inputs.size(0), -1)

        outputs = model(inputs)             # Forward pass
        loss = criterion(outputs, targets)  # Compute loss

        optimizer.zero_grad()               # Zero gradients
        loss.backward()                     # Backward pass
        optimizer.step()                    # Update weights

        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

train(model, data_loader)
```

**Key observation**: The training loop is **identical** whether `num_workers=0` or `num_workers=4`. The parallelism is entirely encapsulated in the `DataLoader`.

### 12.4 CIFAR-10 Batch Arithmetic

```
60,000 images ÷ 64 per batch = 938 batches (937 × 64 + 1 × 32)

Each batch shape:
  inputs:  (64, 3, 32, 32)    ← 64 images, 3 RGB channels, 32×32 pixels
  targets: (64,)              ← 64 integer labels (0-9)

After flattening:
  inputs:  (64, 3072)         ← 64 images, each 3072 values
```

---

## 13. Glossary

| Term | Definition |
|------|-----------|
| **Dataset** | Abstract base class (`torch.utils.data.Dataset`) requiring `__len__` and `__getitem__` |
| **DataLoader** | Wraps a Dataset to provide batching, shuffling, parallel loading, and collation |
| **ETL** | Extract, Transform, Load — the three-phase data pipeline pattern |
| **Extract** | Phase where raw data is read from storage (disk, network, database) |
| **Transform** | Phase where raw data is converted to model-ready format (resize, normalize, augment) |
| **Load** | Phase where prepared data is batched and fed into the model for training |
| **torchvision** | PyTorch domain library for computer vision (datasets, transforms, models) |
| **torchtext** | PyTorch domain library for NLP (deprecated since 2023 — use Hugging Face instead) |
| **torchaudio** | PyTorch domain library for audio processing |
| **ImageFolder** | Dataset class that loads images from subdirectories, using directory names as labels |
| **DatasetFolder** | General-purpose version of ImageFolder for any file type |
| **FakeData** | Dataset class that generates random noise images for prototyping |
| **random_split** | Function that divides a dataset into non-overlapping subsets of specified sizes |
| **batch_size** | Number of samples processed together in one forward/backward pass |
| **shuffle** | Randomize sample order each epoch to prevent learning data ordering |
| **num_workers** | Number of parallel subprocesses for loading data |
| **pin_memory** | Allocate data in page-locked CPU memory for faster GPU transfer |
| **drop_last** | Discard the final incomplete batch if dataset size isn't divisible by batch_size |
| **Collation** | The process of stacking individual samples into batch tensors |
| **Pipelining** | Overlapping data preparation (CPU) with model training (GPU) in time |
| **Corpus** | The body of text used to train a model or build a tokenizer |
| **Sampler** | Controls the order in which samples are drawn from a dataset |

---

## 14. Summary

1. **All PyTorch datasets** inherit from `torch.utils.data.Dataset`, requiring only `__len__` and `__getitem__`
2. **Domain libraries** (torchvision, torchtext, torchaudio) provide pre-built datasets with consistent APIs
3. **Custom datasets** follow a simple pattern: store data/paths in `__init__`, return one sample in `__getitem__`
4. **Generic dataset classes**:
   - `ImageFolder` — loads images from labeled subdirectories (alphabetical class ordering)
   - `DatasetFolder` — same but for any file type
   - `FakeData` — random noise for prototyping
5. **`random_split`** divides a dataset into arbitrary train/val/test ratios without duplication
6. **ETL** (Extract, Transform, Load) is the core data pipeline pattern:
   - Extract: `Dataset` classes read from disk/network
   - Transform: `transforms.Compose(...)` preprocesses each sample
   - Load: `DataLoader` batches, shuffles, and delivers to the model
7. **DataLoader** provides batching, shuffling, parallel loading, and collation
8. **Pipelining** overlaps CPU data preparation with GPU training for maximum throughput
9. **`num_workers`** controls parallelism — start with 2-4 and adjust based on hardware

> **The central insight of this chapter**: Regardless of whether your data is 6 numbers or 6 billion images, the API pattern is the same: `Dataset → Transform → DataLoader → Training Loop`. Master this pattern and you can work with any data at any scale.

---

## 15. Review Questions

1. **What two methods must every PyTorch `Dataset` subclass implement? What does each return?**

2. **You have a directory of images organized as `data/cats/`, `data/dogs/`, `data/birds/`. Which PyTorch class loads this automatically? What label index does each class get?**

3. **You have 50,000 samples and want a 60/20/20 train/val/test split. Write the `random_split` code. How do you ensure no samples are wasted?**

4. **Map the following code to ETL phases — label each line as E, T, or L:**
   ```python
   transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(...)])
   dataset = datasets.CIFAR10(root='./data', download=True, transform=transform)
   loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
   ```

5. **Why is shuffling critical for training but unnecessary for testing?**

6. **Explain the pipelining diagram: why does `num_workers > 0` reduce GPU idle time?**

7. **What does `pin_memory=True` do, and when should you use it?**

8. **Your training is slow. GPU utilization (checked with `nvidia-smi`) is only 30%. The GPU spends most of its time waiting. What DataLoader parameter should you adjust first?**

9. **Why does `ImageFolder` assign classes in alphabetical order? How can this cause bugs, and how do you fix it?**

10. **Write a custom `Dataset` class for a CSV file with two columns: "text" and "label". The `__getitem__` should return `(text_string, label_int)`.** *(Hint: load the CSV in `__init__`, return one row in `__getitem__`.)*

---

> **Phase 3 Complete.** Say **"continue"** to proceed to **Phase 4: Natural Language Processing — Text to Numbers** (Chapters 5-8).
