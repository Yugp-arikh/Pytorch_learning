# Phase 2: Computer Vision — From Dense Networks to Convolutional Neural Networks

> **Covers**: Chapters 2 & 3 of *AI and ML for Coders in PyTorch* by Laurence Moroney
> **Prerequisites**: Phase 1 (tensors, linear layers, MSE, gradient descent, training loop)
> **Goal**: Build image classifiers using dense networks and CNNs; understand every layer, every activation function, every loss function, and every architectural decision — including all underlying mathematics.

---

## Table of Contents

1. [What is Computer Vision?](#1-what-is-computer-vision)
2. [The Fashion MNIST Dataset](#2-the-fashion-mnist-dataset)
3. [From Pixels to Neurons — How a Dense Network Sees Images](#3-from-pixels-to-neurons--how-a-dense-network-sees-images)
4. [Designing the Neural Network Architecture](#4-designing-the-neural-network-architecture)
5. [Activation Functions — ReLU, Softmax, LogSoftmax, Sigmoid](#5-activation-functions--relu-softmax-logsoftmax-sigmoid)
6. [Loss Functions — NLLLoss, CrossEntropyLoss, BCELoss](#6-loss-functions--nllloss-crossentropyloss-bceloss)
7. [Optimizers — Adam vs SGD](#7-optimizers--adam-vs-sgd)
8. [The Complete Fashion MNIST Code — Line by Line](#8-the-complete-fashion-mnist-code--line-by-line)
9. [Training, Evaluation, and Accuracy](#9-training-evaluation-and-accuracy)
10. [Overfitting and Early Stopping](#10-overfitting-and-early-stopping)
11. [Convolutions — The Mathematics of Feature Detection](#11-convolutions--the-mathematics-of-feature-detection)
12. [Pooling — Dimensionality Reduction While Preserving Features](#12-pooling--dimensionality-reduction-while-preserving-features)
13. [Building a CNN in PyTorch — Full Architecture](#13-building-a-cnn-in-pytorch--full-architecture)
14. [Parameter Counting — How Many Numbers Does the Network Learn?](#14-parameter-counting--how-many-numbers-does-the-network-learn)
15. [The Horses or Humans CNN — A Real-World Binary Classifier](#15-the-horses-or-humans-cnn--a-real-world-binary-classifier)
16. [Training, Validation, and Test Sets — The Three-Way Split](#16-training-validation-and-test-sets--the-three-way-split)
17. [Image Augmentation — Virtually Expanding Your Dataset](#17-image-augmentation--virtually-expanding-your-dataset)
18. [Transfer Learning — Standing on the Shoulders of Giants](#18-transfer-learning--standing-on-the-shoulders-of-giants)
19. [Multiclass Classification — Beyond Binary](#19-multiclass-classification--beyond-binary)
20. [Dropout Regularization — Preventing Co-Adaptation](#20-dropout-regularization--preventing-co-adaptation)
21. [Complete Glossary](#21-complete-glossary)
22. [Summary](#22-summary)
23. [Review Questions](#23-review-questions)

---

## 1. What is Computer Vision?

**Computer vision** is a subfield of AI that gives machines the ability to interpret and understand visual information from the world — images, videos, and camera feeds — the way humans do.

### Why It's Hard

Consider a collection of clothing items: shirts, coats, dresses, shoes. You know what makes a shoe a shoe — but try to write that down as `if-else` rules. Two shoes can look completely different (a sneaker vs. a high heel), yet you instantly recognize both as shoes. Now imagine writing rules to distinguish a coat from a dress from a shirt in every possible lighting, angle, color, and background.

**You can't.** The rules are too complex and too numerous to enumerate by hand.

### The ML Approach

Instead of writing rules, we:
1. Collect many images of each category
2. Label them (this is a shoe, this is a shirt...)
3. Train a neural network to discover what distinguishes each category
4. The network learns **features** — visual patterns that define each class

This chapter builds two types of networks:
- **Dense Neural Network (DNN)**: Feeds raw pixels into fully-connected layers
- **Convolutional Neural Network (CNN)**: First extracts features with learned filters, then classifies those features

---

## 2. The Fashion MNIST Dataset

### Background

The original **MNIST** (Modified National Institute of Standards and Technology) dataset was created by Yann LeCun, Corinna Cortes, and Christopher Burges. It contains 70,000 grayscale images of handwritten digits (0-9), each 28x28 pixels. It's the "Hello World" of computer vision.

**Fashion MNIST** is a drop-in replacement created by Zalando Research with the exact same format but clothing items instead of digits:

| Property | Value |
|----------|-------|
| Total images | 70,000 |
| Training set | 60,000 |
| Test set | 10,000 |
| Image size | 28 x 28 pixels |
| Color | Grayscale (1 channel) |
| Pixel values | Integers 0-255 (0 = black, 255 = white) |
| Number of classes | 10 |

### The 10 Classes

| Label | Class Name |
|-------|-----------|
| 0 | T-shirt/Top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle Boot |

### What Each Image Looks Like (Data Representation)

An image is a 28x28 grid. Each cell holds one integer from 0 to 255:

```
Pixel grid (zoomed in on a corner):
  0    0    0    0    0   ...
  0    0    0   64  128   ...
  0    0   48  192  144   ...
  0   142  226  168   ...
  ...
```

**Total values per image**: 28 x 28 = **784 numbers**.

Each of these 784 numbers becomes an input to our neural network.

---

## 3. From Pixels to Neurons — How a Dense Network Sees Images

### The Connection to Chapter 1

In Chapter 1, we had:
- **1 input** (x) → **1 neuron** → **1 output** (y)
- The neuron learned `y = Wx + b` (2 parameters)

Now we have:
- **784 inputs** (one per pixel) → **128 neurons** (hidden layer) → **10 outputs** (one per class)

### How It Works Conceptually

```
INPUT LAYER        HIDDEN LAYER         OUTPUT LAYER
(784 pixels)       (128 neurons)        (10 neurons)

pixel_0  ──┐
pixel_1  ──┤
pixel_2  ──┤──→ neuron_0 ──┐
   ...     │       ...      │──→ class_0 (T-shirt)
pixel_783──┘──→ neuron_127──┤──→ class_1 (Trouser)
                            │       ...
                            └──→ class_9 (Ankle Boot)
```

**Every input pixel is connected to every hidden neuron.** That's why these layers are called **fully-connected** or **dense** layers.

### What Each Hidden Neuron Computes

Each neuron `j` in the hidden layer computes:

```
z_j = W_{j,0}·x_0 + W_{j,1}·x_1 + W_{j,2}·x_2 + ... + W_{j,783}·x_783 + b_j
```

In compact vector notation:

```
z_j = W_j · x + b_j
```

Where:
- **x** is the vector of 784 pixel values (after normalization, so values are 0.0 to 1.0)
- **W_j** is a vector of 784 weights belonging to neuron j
- **b_j** is the bias of neuron j
- **z_j** is the "pre-activation" output (before the activation function is applied)

After computing z_j, we apply an **activation function** (ReLU in our case):

```
a_j = ReLU(z_j) = max(0, z_j)
```

### Why 128 Neurons?

The number 128 is a **hyperparameter** — a design choice made by the programmer, not learned by the network. There is no formula to determine the "perfect" number. Instead:

- **Too few neurons** → The network doesn't have enough capacity to learn the patterns
- **Too many neurons** → The network trains slowly and may memorize the training data (overfitting)
- **128** → A reasonable starting point for a dataset of this complexity

The process of finding good hyperparameters is called **hyperparameter tuning**.

> **Key distinction**: **Parameters** (weights, biases) are learned during training. **Hyperparameters** (number of neurons, learning rate, number of epochs) are chosen by you before training begins.

### Why Flatten the Image?

A Fashion MNIST image is naturally a 2D grid of shape `(28, 28)`. But `nn.Linear` expects a 1D vector as input. So we must **flatten** the image:

```
Before flattening:                After flattening:
┌──────────────────┐              ┌─────────────────────────────┐
│ 28x28 2D grid    │   ──────>    │ 784-element 1D vector       │
│ [[0, 64, 128...] │              │ [0, 64, 128, ..., 168, 0]  │
│  [48, 192, ...]  │              └─────────────────────────────┘
│  ...]            │
└──────────────────┘
```

In PyTorch: `nn.Flatten()` does this automatically.

> **Important limitation**: Flattening destroys spatial information. The network doesn't know that pixel (0,0) is next to pixel (0,1). It just sees 784 independent numbers. This is the main reason dense networks are limited for vision — and why we need CNNs (covered in Sections 11-14).

---

## 4. Designing the Neural Network Architecture

### The Code

```python
class FashionMNISTModel(nn.Module):
    def __init__(self):
        super(FashionMNISTModel, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 128),     # Hidden layer: 784 → 128
            nn.ReLU(),                 # Activation function
            nn.Linear(128, 10),        # Output layer: 128 → 10
            nn.LogSoftmax(dim=1)       # Convert to log-probabilities
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
```

### Anatomy of `nn.Module`

Every neural network in PyTorch is built by subclassing `nn.Module`. This gives you:

1. **`__init__`**: Define all your layers here as class attributes
2. **`forward(self, x)`**: Define how data flows through those layers
3. **Automatic backpropagation**: PyTorch builds a computation graph and computes gradients for you

**`super().__init__()`** is required — it initializes the parent `nn.Module` class, which sets up internal bookkeeping for parameters, gradients, and device management.

### Layer-by-Layer Breakdown

**Layer 1: `nn.Flatten()`**

Converts input shape `(batch_size, 1, 28, 28)` → `(batch_size, 784)`.

**Layer 2: `nn.Linear(28*28, 128)`**

A fully-connected layer:
- Input: 784 values (the flattened pixels)
- Output: 128 values (one per hidden neuron)
- Learnable parameters: 784 × 128 weights + 128 biases = **100,480 parameters**

Mathematically:
```
z = x · W^T + b

Where:
  x  has shape (batch_size, 784)
  W  has shape (128, 784)
  W^T has shape (784, 128)
  b  has shape (128,)
  z  has shape (batch_size, 128)
```

**Layer 3: `nn.ReLU()`**

Applies ReLU activation element-wise (see Section 5 for full math).

**Layer 4: `nn.Linear(128, 10)`**

- Input: 128 values (from hidden layer)
- Output: 10 values (one per clothing class)
- Learnable parameters: 128 × 10 weights + 10 biases = **1,290 parameters**

**Layer 5: `nn.LogSoftmax(dim=1)`**

Converts the 10 raw outputs into **log-probabilities** (see Section 5 for full math).

### Total Parameters

```
Layer               Parameters
─────────────────────────────
Linear(784, 128)    784×128 + 128 = 100,480
Linear(128, 10)     128×10 + 10   = 1,290
─────────────────────────────
TOTAL                             101,770
```

### The `forward` Method

```python
def forward(self, x):
    x = self.flatten(x)                  # (batch, 1, 28, 28) → (batch, 784)
    logits = self.linear_relu_stack(x)   # (batch, 784) → (batch, 10)
    return logits
```

**Forward pass** = the data's journey through the network from input to output. This is where computation happens. PyTorch automatically records every operation to build the computation graph for backpropagation.

**Logits** = the output of the final layer. These are log-probabilities (because we applied LogSoftmax). "Logit" historically means the inverse of the sigmoid function, but in ML, the term is loosely used for "raw model outputs before or after a final transformation."

---

## 5. Activation Functions — ReLU, Softmax, LogSoftmax, Sigmoid

Activation functions are applied after a linear transformation to introduce **non-linearity**. Without them, stacking linear layers would just produce another linear function (a composition of linear functions is still linear). Activation functions let neural networks learn curved, complex decision boundaries.

### 5.1 ReLU (Rectified Linear Unit)

**Definition**:
```
ReLU(z) = max(0, z)
```

**Behavior**:
- If z > 0: output = z (pass through unchanged)
- If z ≤ 0: output = 0 (block the signal)

**Graph**:
```
Output
  │          /
  │         /
  │        /
  │       /
  │      /
──┼─────/────── Input
  │    0
  │
```

**Derivative** (needed for backpropagation):
```
d/dz ReLU(z) = { 1  if z > 0
               { 0  if z < 0
               { undefined at z = 0 (by convention, set to 0 or 1)
```

**Why ReLU?**
1. **Simple and fast** — just a comparison operation
2. **No vanishing gradient problem** — for positive values, the gradient is always 1 (unlike sigmoid/tanh, which saturate and produce near-zero gradients for large inputs)
3. **Sparse activation** — many neurons output 0, which creates sparsity and can improve computational efficiency
4. **Works well in practice** — the most widely used activation in hidden layers

**When to use**: Almost always in hidden (middle) layers. Not for output layers.

**Limitation**: "Dying ReLU" — if a neuron's weights cause it to always produce negative inputs, it will always output 0 and never recover. Variants like LeakyReLU address this.

---

### 5.2 Softmax

**Purpose**: Convert a vector of arbitrary real numbers into a **probability distribution** (all values between 0 and 1, summing to exactly 1).

**Definition**: For a vector z of length K (K = number of classes):

```
Softmax(zᵢ) = e^(zᵢ) / Σⱼ₌₁ᴷ e^(zⱼ)
```

**Worked example**: Suppose the output layer produces `z = [2.0, 1.0, 0.1]` for 3 classes:

```
Step 1: Exponentiate each value
  e^2.0 = 7.389
  e^1.0 = 2.718
  e^0.1 = 1.105

Step 2: Sum them
  7.389 + 2.718 + 1.105 = 11.212

Step 3: Divide each by the sum
  Softmax(z₀) = 7.389 / 11.212 = 0.659
  Softmax(z₁) = 2.718 / 11.212 = 0.242
  Softmax(z₂) = 1.105 / 11.212 = 0.099

Result: [0.659, 0.242, 0.099]
```

**Check**: 0.659 + 0.242 + 0.099 = 1.000 ✓

**Properties**:
- Output is always in (0, 1) — never exactly 0 or 1
- Outputs always sum to 1
- Preserves ordering — the largest input becomes the largest probability
- Amplifies differences — the exponential makes larger values proportionally much larger

**Derivative** (for reference — PyTorch handles this):
```
∂Softmax(zᵢ)/∂zⱼ = Softmax(zᵢ) · (δᵢⱼ - Softmax(zⱼ))

Where δᵢⱼ is the Kronecker delta (1 if i=j, 0 otherwise)
```

---

### 5.3 LogSoftmax

**Definition**: Simply the log of the Softmax:

```
LogSoftmax(zᵢ) = log(Softmax(zᵢ))
                = log(e^(zᵢ) / Σⱼ e^(zⱼ))
                = zᵢ - log(Σⱼ e^(zⱼ))
```

**Why use LogSoftmax instead of Softmax?**

1. **Numerical stability**: Computing `e^z` for large z can overflow. Computing `log(e^z / sum)` directly avoids this.
2. **Pairs with NLLLoss**: PyTorch's `nn.NLLLoss` expects log-probabilities, not raw probabilities. The combination `LogSoftmax + NLLLoss` is mathematically identical to `CrossEntropyLoss` but split into two steps.
3. **Better gradients**: The log function "stretches" small probabilities, preventing the gradient from vanishing when the model is very confident in the wrong answer.

**Worked example**: Using the same z = [2.0, 1.0, 0.1]:

```
LogSoftmax(z₀) = log(0.659) = -0.417
LogSoftmax(z₁) = log(0.242) = -1.417
LogSoftmax(z₂) = log(0.099) = -2.317
```

Note: All values are **negative** (because log of a number between 0 and 1 is always negative). The **closest to 0** is the most confident prediction. In the Fashion MNIST code, when we see output like:

```
tensor([[-12.43, -16.06, -14.31, ..., -1.31, -8.99, -0.33]])
```

The value **-0.33** (closest to 0) indicates the strongest prediction — that's class 9 (Ankle Boot).

---

### 5.4 Sigmoid

**Definition**:
```
σ(z) = 1 / (1 + e^(-z))
```

**Behavior**:
- As z → +∞: σ(z) → 1
- As z → -∞: σ(z) → 0
- At z = 0: σ(z) = 0.5

**Graph**:
```
Output
 1.0  ─────────────────────────
                          ╱
 0.5  ─ ─ ─ ─ ─ ─ ─ ─ ─╱─ ─ ─
                       ╱
 0.0  ─────────────────
      ──────────────────────── Input
              -5   0   5
```

**Derivative**:
```
dσ/dz = σ(z) · (1 - σ(z))
```

This is elegant — the derivative is expressed entirely in terms of the function itself.

**When to use**: Output layer for **binary classification** — the sigmoid squashes any real number into (0, 1), which we interpret as the probability of one class.

**Example**: If sigmoid outputs 0.82 for a horse-or-human classifier:
- P(Human) = 0.82
- P(Horse) = 1 - 0.82 = 0.18
- Prediction: Human (threshold at 0.5)

**Why not in hidden layers?**
The sigmoid has a **vanishing gradient problem**: for very large or very small z, the derivative `σ(z)(1-σ(z))` is nearly 0. This means gradients barely flow backward through the network, making deep networks very hard to train. ReLU solves this.

---

## 6. Loss Functions — NLLLoss, CrossEntropyLoss, BCELoss

### 6.1 Negative Log-Likelihood Loss (NLLLoss)

**Used when**: Your model outputs **log-probabilities** (i.e., you used `LogSoftmax` as the last activation).

**Definition**: Given:
- **ŷ** = vector of log-probabilities output by the model (one per class)
- **c** = the correct class index (ground truth label)

```
NLLLoss = -ŷ_c
```

That's it. It simply takes the **negative** of the log-probability assigned to the **correct** class.

**Intuition**: If the model assigns log-probability -0.1 to the correct class, the loss is 0.1 (very low — the model is confident and correct). If it assigns -5.0, the loss is 5.0 (high — the model is not confident in the correct answer).

**Worked example**: Model output (log-probabilities) = [-2.3, -0.1, -4.5] for 3 classes. True label = class 1.

```
NLLLoss = -(-0.1) = 0.1
```

The model is confident about class 1, so loss is low. ✓

What if the true label were class 2?

```
NLLLoss = -(-4.5) = 4.5
```

The model gave very low probability to class 2, so loss is high. ✓

**In batch form** (averaging over a batch of N samples):

```
NLLLoss = -(1/N) · Σᵢ₌₁ᴺ ŷᵢ,cᵢ
```

Where cᵢ is the correct class for sample i and ŷᵢ,cᵢ is the log-probability the model assigned to that class.

---

### 6.2 Cross-Entropy Loss (CrossEntropyLoss)

**Mathematically equivalent** to `LogSoftmax + NLLLoss`, but combined into one function for convenience and numerical stability.

**Definition**: Given raw output scores z (before any activation) and correct class c:

```
CrossEntropyLoss = -log(Softmax(z_c))
                 = -log(e^(z_c) / Σⱼ e^(z_j))
                 = -z_c + log(Σⱼ e^(z_j))
```

**When to use**: Multi-class classification. You do NOT need LogSoftmax on your output layer — CrossEntropyLoss handles it internally.

**Relationship**:
```
LogSoftmax + NLLLoss  ≡  CrossEntropyLoss
```

Both produce the same gradients and the same training behavior. Choose based on style:
- If you want to inspect log-probabilities from your model → use LogSoftmax + NLLLoss
- If you want simplicity → use CrossEntropyLoss directly on raw scores

---

### 6.3 Binary Cross-Entropy Loss (BCELoss)

**Used when**: You have **exactly 2 classes** and a **single output neuron** with sigmoid activation.

**Definition**: Given:
- **ŷ** = sigmoid output (a single probability in [0, 1])
- **y** = true label (0 or 1)

```
BCELoss = -[y · log(ŷ) + (1-y) · log(1-ŷ)]
```

**How it works**:
- When y = 1 (true class is positive): Loss = -log(ŷ). If ŷ is close to 1, loss is near 0. If ŷ is close to 0, loss is very high.
- When y = 0 (true class is negative): Loss = -log(1-ŷ). If ŷ is close to 0, loss is near 0. If ŷ is close to 1, loss is very high.

**Worked example**: True label y = 1, model predicts ŷ = 0.9:
```
BCELoss = -[1·log(0.9) + 0·log(0.1)]
        = -log(0.9)
        = -(-0.105)
        = 0.105     ← low loss, good prediction ✓
```

True label y = 1, model predicts ŷ = 0.1:
```
BCELoss = -[1·log(0.1) + 0·log(0.9)]
        = -log(0.1)
        = -(-2.303)
        = 2.303     ← high loss, bad prediction ✓
```

**Batch form**:
```
BCELoss = -(1/N) · Σᵢ₌₁ᴺ [yᵢ·log(ŷᵢ) + (1-yᵢ)·log(1-ŷᵢ)]
```

### Summary Table: Which Loss Function to Use

| Scenario | Output Activation | Loss Function | Output Neurons |
|----------|------------------|---------------|---------------|
| Binary classification (2 classes) | Sigmoid | `nn.BCELoss()` | 1 |
| Multi-class (using LogSoftmax) | LogSoftmax | `nn.NLLLoss()` | N (one per class) |
| Multi-class (using raw scores) | None | `nn.CrossEntropyLoss()` | N (one per class) |

---

## 7. Optimizers — Adam vs SGD

### 7.1 SGD (Stochastic Gradient Descent) — Recap from Phase 1

```
W_new = W_old - α · ∂L/∂W
```

Simple, well-understood, but:
- Uses a **fixed learning rate** for all parameters
- Can oscillate in steep valleys
- Can be slow to converge on flat surfaces

### 7.2 Adam (Adaptive Moment Estimation)

Adam combines two ideas:

**Idea 1 — Momentum**: Keep a running average of past gradients. Instead of updating based on just the current gradient, use a smoothed version. This dampens oscillations.

```
m_t = β₁ · m_{t-1} + (1 - β₁) · g_t          (first moment / mean of gradients)
```

**Idea 2 — RMSprop**: Keep a running average of the **squared** gradients. Use this to adapt the learning rate per-parameter. Parameters with consistently large gradients get smaller steps; parameters with small gradients get larger steps.

```
v_t = β₂ · v_{t-1} + (1 - β₂) · g_t²         (second moment / variance of gradients)
```

**Bias correction** (important at the start of training when m and v are initialized to 0):

```
m̂_t = m_t / (1 - β₁^t)
v̂_t = v_t / (1 - β₂^t)
```

**Parameter update**:

```
W_new = W_old - α · m̂_t / (√v̂_t + ε)
```

Where:
- **α** = learning rate (default 0.001)
- **β₁** = decay rate for first moment (default 0.9)
- **β₂** = decay rate for second moment (default 0.999)
- **ε** = small constant to prevent division by zero (default 10⁻⁸)
- **g_t** = gradient at step t

**Why Adam is generally preferred over SGD**:
- Adapts the learning rate per parameter
- Works well with sparse gradients
- Requires less hyperparameter tuning
- Converges faster on most problems

**In code**:
```python
# SGD
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Adam (generally faster and more forgiving)
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

---

## 8. The Complete Fashion MNIST Code — Line by Line

### 8.1 Imports and Data Loading

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
```

| Import | What It Provides |
|--------|-----------------|
| `torch` | Core tensor library |
| `torch.nn` | Layer types, loss functions, model containers |
| `torch.optim` | Optimizer algorithms |
| `torchvision.datasets` | Pre-built datasets (Fashion MNIST, CIFAR, ImageNet, etc.) |
| `torchvision.transforms` | Data preprocessing (resize, normalize, augment) |
| `DataLoader` | Batching, shuffling, parallel data loading |

### 8.2 Transforms and Normalization

```python
transform = transforms.Compose([transforms.ToTensor()])
```

**`transforms.Compose`**: Chains multiple transforms in order. Here, there's only one.

**`transforms.ToTensor()`**: Does two things simultaneously:
1. Converts a PIL Image or NumPy array to a PyTorch tensor
2. Rescales pixel values from **[0, 255] integers** to **[0.0, 1.0] floats**

The rescaling formula is simply:
```
pixel_normalized = pixel_original / 255.0
```

**Why normalize?**
- Neural networks work best with small input values (typically between -1 and 1 or 0 and 1)
- Large input values cause large gradients, which can make training unstable (the loss explodes or oscillates)
- Normalization ensures all features (pixels) are on the same scale

### 8.3 Loading the Datasets

```python
train_dataset = datasets.FashionMNIST(root='./data', train=True,
                                       download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False,
                                      download=True, transform=transform)
```

| Parameter | Meaning |
|-----------|---------|
| `root='./data'` | Directory to store the downloaded data |
| `train=True` | Load the 60,000-image training set |
| `train=False` | Load the 10,000-image test set |
| `download=True` | Download if not already present |
| `transform=transform` | Apply the ToTensor normalization |

**Why two separate datasets?** We train on one set and evaluate on a different set. If we only used one set, we'd have no way to know if the model truly learned generalizable patterns or just memorized the training data.

### 8.4 DataLoaders — Batching and Shuffling

```python
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
```

**What a DataLoader does**:
1. **Batching**: Groups samples into batches. Instead of feeding all 60,000 images at once (too much memory) or one at a time (too slow), we feed 64 at a time.
2. **Shuffling**: Randomizes the order of samples each epoch. This prevents the network from learning the order of the data.
3. **Parallel loading**: Can use multiple workers to load data in parallel with training.

**Batch arithmetic for Fashion MNIST**:
```
60,000 images ÷ 64 per batch = 937.5 → 938 batches
  (937 batches of 64 + 1 batch of 32)
```

**A single batch has shape**: `(64, 1, 28, 28)` — 64 images, 1 channel (grayscale), 28 height, 28 width.

**Why shuffle training but not test?**
- Training: shuffling prevents the model from memorizing patterns in the data order
- Testing: order doesn't matter (we just want accuracy), and no-shuffle makes results reproducible

### 8.5 The Model Class

```python
class FashionMNISTModel(nn.Module):
    def __init__(self):
        super(FashionMNISTModel, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = FashionMNISTModel()
```

Data flow for a single image:

```
Input:    (1, 28, 28)    ← 1 channel, 28x28 pixels
          ↓ Flatten
          (784,)          ← 1D vector of 784 values
          ↓ Linear(784→128)
          (128,)          ← 128 pre-activation values
          ↓ ReLU
          (128,)          ← 128 post-activation values (negatives zeroed)
          ↓ Linear(128→10)
          (10,)           ← 10 raw scores
          ↓ LogSoftmax
          (10,)           ← 10 log-probabilities
```

### 8.6 Loss and Optimizer

```python
loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters())
```

- **NLLLoss** pairs with **LogSoftmax** (as explained in Section 6.1)
- **Adam** is used with default learning rate (0.001)
- **`model.parameters()`** passes all learnable parameters (100,480 + 1,290 = 101,770 values) to the optimizer

---

## 9. Training, Evaluation, and Accuracy

### 9.1 The Training Function

```python
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)        # 60,000
    model.train()                          # Set model to training mode
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)                    # Forward pass
        loss = loss_fn(pred, y)            # Compute loss

        optimizer.zero_grad()              # Zero gradients
        loss.backward()                    # Backward pass
        optimizer.step()                   # Update weights

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
```

**`model.train()`**: Sets the model to training mode. This matters because certain layers behave differently during training vs. evaluation:
- **Dropout**: Active during training (randomly zeroes neurons), disabled during evaluation
- **BatchNorm**: Uses batch statistics during training, running statistics during evaluation

**`enumerate(dataloader)`**: Returns `(batch_index, (images, labels))` for each batch.

**`loss.item()`**: Extracts the loss value as a plain Python float (detached from the computation graph).

### 9.2 The Evaluation Function

```python
def test(dataloader, model):
    size = len(dataloader.dataset)         # 10,000
    num_batches = len(dataloader)
    model.eval()                           # Set model to evaluation mode
    test_loss, correct = 0, 0
    with torch.no_grad():                  # Disable gradient computation
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_function(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")
```

**`model.eval()`**: Sets model to evaluation mode (disables dropout, etc.).

**`torch.no_grad()`**: Disables gradient tracking. During evaluation, we don't need gradients (no backpropagation), so this saves memory and speeds up computation.

**`pred.argmax(1)`**: Critical operation. Let's break it down:

The model outputs shape `(batch_size, 10)` — 10 log-probabilities per image. `argmax(1)` finds the index of the maximum value along dimension 1 (the class dimension).

```
Example for one image:
pred = [-12.4, -16.1, -14.3, -16.3, -13.2, -4.5, -13.6, -1.3, -9.0, -0.3]
argmax = 9  (because -0.3 is the largest value)
Label 9 = Ankle Boot ✓
```

**`(pred.argmax(1) == y)`**: Creates a boolean tensor — True where prediction matches label.

**`.type(torch.float).sum().item()`**: Converts True/False to 1.0/0.0, sums them, extracts as Python number.

### 9.3 The Training Loop

```python
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_loader, model, loss_function, optimizer)
print("Done!")
```

### 9.4 Typical Results

| Epoch | Training Accuracy | Test Accuracy | Observation |
|-------|------------------|---------------|-------------|
| 1 | ~83% | ~84% | Network is learning fast |
| 3 | ~87% | ~86% | Improvement slowing down |
| 5 | ~88.6% | ~86.9% | Gap widening (sign of overfitting) |

**Key observation**: Training accuracy (88.6%) > Test accuracy (86.9%). This gap indicates the network has started memorizing training data rather than generalizing. This is **overfitting**.

### 9.5 Computing Accuracy During Training

```python
def get_accuracy(pred, labels):
    _, predictions = torch.max(pred, 1)   # Get index of max value
    correct = (predictions == labels).float().sum()
    accuracy = correct / labels.shape[0]  # Divide by batch size
    return accuracy
```

**`torch.max(pred, 1)`** returns two tensors:
1. The maximum values (we discard these with `_`)
2. The indices of the maximum values (these are our predicted classes)

---

## 10. Overfitting and Early Stopping

### 10.1 What is Overfitting?

**Overfitting** occurs when a model learns the training data too well — it memorizes specific examples rather than learning general patterns.

```
                 Accuracy
                    │
    Training  ──────┼──────────── ↗ 96.15%
                    │          ↗
                    │        ↗
                    │      ↗ ↗
    Test      ──────┼────↗────── → 89.2%   ← Plateau/slight increase
                    │  ↗
                    │↗
                    └────────────────────── Epochs
                    1        25        50
```

After 50 epochs:
- Training accuracy: **96.15%** (up from 88.6% at epoch 5)
- Test accuracy: **89.2%** (up from 86.9% at epoch 5)

The training accuracy improved by ~8%, but test accuracy only improved by ~2%. The growing gap means the model is overfitting.

**Analogy**: A student who memorizes answers to practice problems but doesn't understand the concepts. They ace the practice test but struggle on the real exam.

### 10.2 Why Does Overfitting Happen?

1. **Too much training** — The model has too many epochs to memorize the data
2. **Too little data** — Not enough examples to learn general patterns
3. **Too many parameters** — The model is too complex for the problem
4. **No regularization** — Nothing prevents the model from memorizing

### 10.3 Early Stopping

Instead of training for a fixed number of epochs, we **stop when performance is good enough**:

```python
if avg_accuracy >= 95:
    print("Reached 95% accuracy, stopping training.")
    return True  # Stop training
```

**Benefits**:
- Saves time (reached 95% at epoch 37 instead of running all 50)
- Prevents overfitting (stopping before the model memorizes too much)
- Automated — no need to manually check and restart

**Best practice**: Monitor **validation accuracy** (not training accuracy) and stop when it stops improving. This is the standard technique in production ML.

---

## 11. Convolutions — The Mathematics of Feature Detection

This is the most important section in Phase 2. Convolutions are the key idea that makes modern computer vision work.

### 11.1 The Problem with Dense Networks for Images

A dense network treats every pixel independently. It doesn't know that pixel (5, 5) is adjacent to pixel (5, 6). It has no concept of spatial relationships, edges, textures, or shapes. It just sees 784 random numbers.

This works for simple datasets (Fashion MNIST with centered items), but fails for complex scenes where objects can appear at any position in the image.

### 11.2 What is a Convolution?

A **convolution** is a mathematical operation that slides a small matrix (called a **filter** or **kernel**) over the image, computing a weighted sum at each position. The result is a new image (called a **feature map**) that highlights certain features.

### 11.3 The Convolution Operation — Step by Step

**Given**: A 3x3 region of an image and a 3x3 filter:

```
Image region:               Filter (kernel):
┌─────┬─────┬─────┐        ┌─────┬─────┬─────┐
│  0  │  64 │ 128 │        │ -1  │  0  │ -2  │
├─────┼─────┼─────┤        ├─────┼─────┼─────┤
│  48 │ 192 │ 144 │        │ 0.5 │ 4.5 │-1.5 │
├─────┼─────┼─────┤        ├─────┼─────┼─────┤
│ 142 │ 226 │ 168 │        │ 1.5 │  2  │ -3  │
└─────┴─────┴─────┘        └─────┴─────┴─────┘
```

**Computation**: Multiply each pixel by the corresponding filter value, then sum:

```
new_value = (-1 × 0)   + (0 × 64)    + (-2 × 128)
          + (0.5 × 48) + (4.5 × 192) + (-1.5 × 144)
          + (1.5 × 142)+ (2 × 226)   + (-3 × 168)

         = 0 + 0 + (-256)
         + 24 + 864 + (-216)
         + 213 + 452 + (-504)

         = 577
```

The original center pixel value was 192. After the convolution, it becomes **577**.

### 11.4 General Formula

For an image **I** and a filter **K** of size (2r+1) × (2r+1), the output at position (x, y) is:

```
                r     r
Output(x,y) =  Σ     Σ   I(x+i, y+j) · K(i, j)
              i=-r  j=-r
```

For a 3×3 filter (r=1):
```
                1     1
Output(x,y) =  Σ     Σ   I(x+i, y+j) · K(i, j)
              i=-1  j=-1
```

### 11.5 Sliding the Filter Across the Entire Image

The filter doesn't stay in one place — it **slides** across every position:

```
Step 1:                 Step 2:                 Step 3:
┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│[■ ■ ■]· · · · │      │ ·[■ ■ ■]· · · │      │ · ·[■ ■ ■]· · │
│[■ ■ ■]· · · · │      │ ·[■ ■ ■]· · · │      │ · ·[■ ■ ■]· · │
│[■ ■ ■]· · · · │      │ ·[■ ■ ■]· · · │      │ · ·[■ ■ ■]· · │
│ · · · · · · · │      │ · · · · · · · │      │ · · · · · · · │
└───────────────┘      └───────────────┘      └───────────────┘
  Produces 1 value       Produces 1 value       Produces 1 value
```

At each position, the filter computes one value → the full output is a new image (feature map).

### 11.6 What Different Filters Do

The magic is that different filter values detect different features:

**Vertical edge detector**:
```
Filter:          Effect:
┌────┬───┬────┐  Positive values on the right, negative on
│ -1 │ 0 │  1 │  the left → detects vertical edges (where
├────┼───┼────┤  brightness changes left-to-right)
│ -1 │ 0 │  1 │
├────┼───┼────┤
│ -1 │ 0 │  1 │
└────┴───┴────┘
```

**Horizontal edge detector**:
```
Filter:          Effect:
┌────┬────┬────┐  Positive values at bottom, negative at
│ -1 │ -1 │ -1 │  top → detects horizontal edges
├────┼────┼────┤
│  0 │  0 │  0 │
├────┼────┼────┤
│  1 │  1 │  1 │
└────┴────┴────┘
```

**Sharpen**:
```
┌────┬────┬────┐
│  0 │ -1 │  0 │
├────┼────┼────┤
│ -1 │  5 │ -1 │
├────┼────┼────┤
│  0 │ -1 │  0 │
└────┴────┴────┘
```

> **The key insight for CNNs**: In a CNN, the filter values are **not hand-designed**. They are **learned** during training. The network discovers which filters best help it distinguish between classes.

### 11.7 Border Effects and Padding

When a 3×3 filter slides over a 28×28 image, it can't center on edge pixels (there aren't enough neighbors). The first valid center position is (1,1), the last is (26,26).

**Result**: Output image is (28-2) × (28-2) = **26 × 26**. We lose 1 pixel on each side.

**General formula**: For image size `n × n` and filter size `f × f`:
```
Output size = (n - f + 1) × (n - f + 1)
```

For 28×28 image with 3×3 filter: (28 - 3 + 1) = 26 → output is 26×26.

**Padding** solves this by adding zeros around the border:

```
Without padding (28×28 → 26×26):       With padding=1 (28×28 → 28×28):
┌──────────────────┐                    ┌──0─0─0─0─0─0──0──┐  ← zero border
│  Image 28×28     │                    │ 0[Image 28×28  ]0 │
│  ┌────────────┐  │                    │ 0[              ]0 │
│  │Output 26×26│  │                    │ 0[Output 28×28  ]0 │
│  └────────────┘  │                    │ 0[              ]0 │
└──────────────────┘                    └──0─0─0─0─0─0──0──┘
```

**With padding `p`**, the output size becomes:
```
Output size = (n + 2p - f + 1) × (n + 2p - f + 1)
```

For padding=1, 28×28 image, 3×3 filter: (28 + 2 - 3 + 1) = 28 → output stays 28×28.

### 11.8 Stride

**Stride** = how many pixels the filter moves at each step. Default is 1 (move one pixel at a time).

With stride `s`, the output size is:
```
Output size = floor((n + 2p - f) / s + 1)
```

Example: 28×28 image, 3×3 filter, padding=1, stride=2:
```
Output = floor((28 + 2 - 3) / 2 + 1) = floor(13.5 + 1) = 14 × 14
```

### 11.9 Multiple Channels

Color images have 3 channels (Red, Green, Blue). The filter must also have 3 channels:

```
Input:           3 channels × 28 × 28
Filter:          3 channels × 3 × 3
```

The filter slides over all 3 channels simultaneously, and the results are summed into one value:

```
Output(x,y) = Σ_c Σ_i Σ_j Input(c, x+i, y+j) · Filter(c, i, j) + bias
```

Where c iterates over channels. One filter → one output feature map (single channel).

### 11.10 Multiple Filters

We typically use many filters (e.g., 64). Each filter produces one feature map. So:

```
Input:  C_in channels × H × W
        ↓
64 filters, each of size C_in × 3 × 3
        ↓
Output: 64 channels × H' × W'
```

The 64 output channels are stacked into one tensor — this becomes the input to the next layer.

---

## 12. Pooling — Dimensionality Reduction While Preserving Features

### 12.1 What is Pooling?

Pooling reduces the spatial dimensions (height and width) of feature maps while retaining the most important information. This:
1. Reduces computational cost
2. Reduces number of parameters (less overfitting)
3. Makes the network more robust to small translations

### 12.2 Max Pooling

The most common type. Divides the feature map into non-overlapping regions and takes the **maximum** value from each.

**Example**: 2×2 max pooling with stride 2:

```
Input (4×4):                    Output (2×2):
┌─────┬─────┬─────┬─────┐      ┌─────┬─────┐
│  1  │  3  │  5  │  0  │      │  4  │  5  │  ← max(1,3,2,4)=4, max(5,0,1,3)=5
├─────┼─────┼─────┼─────┤      ├─────┼─────┤
│  2  │  4  │  1  │  3  │      │  8  │  9  │  ← max(6,8,0,7)=8, max(2,9,4,1)=9
├─────┼─────┼─────┼─────┤      └─────┴─────┘
│  6  │  8  │  2  │  9  │
├─────┼─────┼─────┼─────┤
│  0  │  7  │  4  │  1  │
└─────┴─────┴─────┴─────┘
```

**Formula**: For pool size `k × k` with stride `s`:
```
Output size = floor((Input_size - k) / s + 1)
```

For 14×14 input, pool 2×2, stride 2:
```
Output = floor((14 - 2) / 2 + 1) = 7 × 7
```

**Properties**:
- Has **no learnable parameters** (just a fixed operation)
- Reduces each spatial dimension by a factor of 2 (for 2×2 pooling with stride 2)
- Preserves the most activated features (because it keeps the maximum)

### 12.3 Other Pooling Types

| Type | Operation | Use Case |
|------|----------|----------|
| **Max Pooling** | Takes maximum value in each pool | Most common; preserves strongest activations |
| **Average Pooling** | Takes mean of all values in each pool | Smoother; used in some architectures |
| **Min Pooling** | Takes minimum value | Rarely used |
| **Global Average Pooling** | Averages the entire feature map to a single value | Used before the final layer in some modern architectures |

### 12.4 In PyTorch

```python
nn.MaxPool2d(kernel_size=2, stride=2)  # 2×2 pool, stride 2
```

- `kernel_size=2` → pool regions are 2×2
- `stride=2` → move 2 pixels between pools (non-overlapping)

---

## 13. Building a CNN in PyTorch — Full Architecture

### 13.1 The Fashion MNIST CNN

```python
class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),   # (1,28,28) → (64,28,28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))         # (64,28,28) → (64,14,14)

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),              # (64,14,14) → (64,12,12)
            nn.ReLU(),
            nn.MaxPool2d(2))                               # (64,12,12) → (64,6,6)

        self.fc1 = nn.Linear(64 * 6 * 6, 128)             # 2304 → 128
        self.fc2 = nn.Linear(128, 10)                      # 128 → 10

    def forward(self, x):
        out = self.layer1(x)          # Convolution + ReLU + Pool
        out = self.layer2(out)        # Convolution + ReLU + Pool
        out = out.view(out.size(0), -1)  # Flatten: (batch, 64, 6, 6) → (batch, 2304)
        out = self.fc1(out)           # Dense layer
        out = self.fc2(out)           # Output layer
        return out
```

### 13.2 Data Flow — Shape at Every Step

Let's trace the shape of a single image through the network:

```
STEP    LAYER                       OUTPUT SHAPE        WHAT HAPPENS
─────   ─────────────────────────   ──────────────      ──────────────────────────────
Input   -                           (1, 28, 28)         Grayscale image
  1     Conv2d(1→64, k=3, p=1)     (64, 28, 28)        64 filters, padding preserves size
  2     ReLU                        (64, 28, 28)        Zero out negatives
  3     MaxPool2d(2, 2)             (64, 14, 14)        Halve spatial dimensions
  4     Conv2d(64→64, k=3)         (64, 12, 12)        No padding → lose 2 pixels each side
  5     ReLU                        (64, 12, 12)        Zero out negatives
  6     MaxPool2d(2)                (64, 6, 6)          Halve again
  7     view(-1)                    (2304,)             Flatten: 64 × 6 × 6 = 2304
  8     Linear(2304→128)           (128,)              Dense hidden layer
  9     Linear(128→10)             (10,)               Output: 10 class scores
```

### 13.3 The `view` Operation (Flattening)

```python
out = out.view(out.size(0), -1)
```

- `out.size(0)` = batch size (e.g., 64)
- `-1` = "infer this dimension" → PyTorch calculates `64 × 6 × 6 = 2304`
- Transforms shape from `(batch, 64, 6, 6)` to `(batch, 2304)`

This flattening is necessary because `nn.Linear` expects 1D input per sample.

### 13.4 Results

Training for 50 epochs: **91.31% test accuracy** (compared to 89.2% with the dense network).

The CNN outperforms the DNN because convolutions detect spatial features (edges, textures, shapes) rather than treating each pixel independently.

---

## 14. Parameter Counting — How Many Numbers Does the Network Learn?

Understanding parameter counts is essential for debugging architectures and estimating training time.

### 14.1 Conv2d Parameters

For `nn.Conv2d(C_in, C_out, kernel_size=k)`:

```
Parameters = C_out × (C_in × k × k + 1)
                                    ↑
                                   bias
```

Or equivalently:
```
Weights = C_out × C_in × k × k
Biases  = C_out
Total   = Weights + Biases
```

**Layer 1**: `Conv2d(1, 64, kernel_size=3, padding=1)`
```
Weights = 64 × 1 × 3 × 3 = 576
Biases  = 64
Total   = 640
```

**Layer 2**: `Conv2d(64, 64, kernel_size=3)`
```
Weights = 64 × 64 × 3 × 3 = 36,864
Biases  = 64
Total   = 36,928
```

### 14.2 ReLU and MaxPool Parameters

**Zero.** These layers have no learnable parameters. ReLU is just `max(0, x)` and MaxPool just selects maximum values.

### 14.3 Linear Parameters

For `nn.Linear(in_features, out_features)`:

```
Parameters = in_features × out_features + out_features
           = (in × out) + out
```

**fc1**: `Linear(64 × 6 × 6, 128)` = `Linear(2304, 128)`
```
Parameters = 2304 × 128 + 128 = 295,040
```

**fc2**: `Linear(128, 10)`
```
Parameters = 128 × 10 + 10 = 1,290
```

### 14.4 Full Summary

```
Layer                Output Shape         Parameters
─────────────────    ──────────────       ──────────
Conv2d-1             (64, 28, 28)         640
ReLU-2               (64, 28, 28)         0
MaxPool2d-3          (64, 14, 14)         0
Conv2d-4             (64, 12, 12)         36,928
ReLU-5               (64, 12, 12)         0
MaxPool2d-6          (64, 6, 6)           0
Linear-7             (128)                295,040
Linear-8             (10)                 1,290
─────────────────────────────────────────────────────
TOTAL                                     333,898
```

Compare to the DNN's 101,770 parameters. The CNN has **3x more parameters** but achieves significantly higher accuracy because those parameters are organized to detect spatial features.

> **Note**: Most parameters (295,040 / 333,898 = **88%**) are in the first Linear layer, where the flattened convolution output connects to the dense layer. This is typical — the dense layers dominate parameter count.

---

## 15. The Horses or Humans CNN — A Real-World Binary Classifier

### 15.1 Dataset Properties

| Property | Value |
|----------|-------|
| Image size | 300×300 (resized to 150×150) |
| Color | RGB (3 channels) |
| Training images | ~1,027 |
| Validation images | 256 |
| Classes | 2 (Horse, Human) |
| Content | Computer-generated imagery (CGI) |

### 15.2 Data Pipeline

```python
# Transforms
transform = transforms.Compose([
    transforms.Resize((150, 150)),                              # Resize to 150×150
    transforms.ToTensor(),                                      # [0,255] → [0,1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5],                # Center around 0
                         std=[0.5, 0.5, 0.5])                  # Scale to [-1, 1]
])

# Load from directory structure
train_dataset = datasets.ImageFolder(root=training_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=validation_dir, transform=transform)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
```

**`datasets.ImageFolder`**: Automatically assigns labels based on subdirectory names:
```
training/
├── horses/    ← label 0
│   ├── horse01.png
│   └── horse02.png
└── humans/    ← label 1
    ├── human01.png
    └── human02.png
```

**Normalization formula**: `transforms.Normalize(mean, std)` computes:
```
pixel_out = (pixel_in - mean) / std
```
With mean=0.5, std=0.5: `pixel_out = (pixel_in - 0.5) / 0.5 = 2·pixel_in - 1`

This maps [0, 1] → [-1, 1], centering the data around zero.

### 15.3 The Architecture

```python
class HorsesHumansCNN(nn.Module):
    def __init__(self):
        super(HorsesHumansCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)    # (3,150,150)→(16,150,150)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)   # (16,75,75)→(32,75,75)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)   # (32,37,37)→(64,37,37)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 18 * 18, 512)
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(512, 1)      # Single output neuron

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # → (16,75,75)
        x = self.pool(F.relu(self.conv2(x)))   # → (32,37,37)
        x = self.pool(F.relu(self.conv3(x)))   # → (64,18,18)
        x = x.view(-1, 64 * 18 * 18)          # Flatten: → (20736)
        x = F.relu(self.fc1(x))                # → (512)
        x = self.drop(x)                       # Dropout
        x = self.fc2(x)                        # → (1)
        x = torch.sigmoid(x)                   # → probability in [0, 1]
        return x
```

### 15.4 Shape Trace

```
Input:                          (3, 150, 150)    RGB image
Conv2d(3→16, k=3, p=1)         (16, 150, 150)   16 filters, padding preserves size
MaxPool2d(2,2)                  (16, 75, 75)     Halve: 150/2 = 75
Conv2d(16→32, k=3, p=1)        (32, 75, 75)     32 filters
MaxPool2d(2,2)                  (32, 37, 37)     Halve: 75/2 = 37 (floor)
Conv2d(32→64, k=3, p=1)        (64, 37, 37)     64 filters
MaxPool2d(2,2)                  (64, 18, 18)     Halve: 37/2 = 18 (floor)
Flatten                         (20,736)         64 × 18 × 18
Linear(20736→512)               (512)            Dense hidden
Dropout(0.25)                   (512)            Drop 25% of neurons
Linear(512→1)                   (1)              Single output
Sigmoid                         (1)              Probability: [0, 1]
```

### 15.5 Why One Output Neuron with Sigmoid?

For **binary classification**, you have two choices:

| Approach | Output Neurons | Activation | Loss | Threshold |
|----------|---------------|-----------|------|-----------|
| **Single neuron** | 1 | Sigmoid | BCELoss | > 0.5 = Class 1 |
| **Two neurons** | 2 | Softmax | CrossEntropyLoss | argmax |

Both are valid. This model uses one neuron with sigmoid:
- Output ≈ 0 → Horse
- Output ≈ 1 → Human
- Decision boundary at 0.5

### 15.6 Parameter Count

```
Layer                  Output Shape         Parameters
──────────────────     ──────────────       ──────────
Conv2d(3→16)           (16, 150, 150)       448
MaxPool2d              (16, 75, 75)         0
Conv2d(16→32)          (32, 75, 75)         4,640
MaxPool2d              (32, 37, 37)         0
Conv2d(32→64)          (64, 37, 37)         18,496
MaxPool2d              (64, 18, 18)         0
Linear(20736→512)      (512)                10,617,344
Dropout                (512)                0
Linear(512→1)          (1)                  513
──────────────────────────────────────────────────────
TOTAL                                       10,641,441
```

Over **10 million parameters** — mostly in the first Linear layer (10.6M / 10.6M = 99.8%).

---

## 16. Training, Validation, and Test Sets — The Three-Way Split

### 16.1 Definitions

| Set | Purpose | When Used | Used to Update Weights? |
|-----|---------|-----------|------------------------|
| **Training** | Teach the model — fit data to labels | During training | **Yes** |
| **Validation** | Monitor performance during training; tune hyperparameters | After each epoch | **No** |
| **Test** | Final, unbiased evaluation of the trained model | Once, after all training | **No** |

### 16.2 Why Three Sets?

If you only use train and test:
- You train on the training set
- You evaluate on the test set
- You tweak hyperparameters based on test performance
- You evaluate again on the test set
- **Problem**: You're indirectly "training" on the test set by using it to make architecture decisions

With a validation set:
- You train on the training set
- You evaluate on the **validation** set after each epoch
- You tweak hyperparameters based on validation performance
- The **test** set is only used once, at the very end
- **Result**: The test set gives a truly unbiased estimate of real-world performance

### 16.3 Typical Observations from Horses or Humans

```
Epoch 7:   Training = 100.0%    Validation = 88.3%
Epoch 8:   Training = 100.0%    Validation = 89.1%
Epoch 9:   Training = 100.0%    Validation = 86.3%   ← fluctuating
Epoch 10:  Training = 100.0%    Validation = 87.5%
```

**100% training accuracy with ~87% validation accuracy** = severe overfitting. The model has memorized the ~1,000 training images but doesn't fully generalize to unseen images.

---

## 17. Image Augmentation — Virtually Expanding Your Dataset

### 17.1 The Problem

Small datasets lead to overfitting. Collecting more data is expensive. Can we make the model think it has more data?

### 17.2 The Solution

**Image augmentation** applies random transformations to training images on-the-fly. Each epoch, the model sees slightly different versions of the same images.

### 17.3 Available Transforms

```python
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),       # 50% chance of flipping left-right
    transforms.RandomRotation(20),           # Rotate ±20 degrees
    transforms.RandomResizedCrop(150),       # Random crop and resize to 150×150
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
```

**Additional transforms via `RandomAffine`**:
```python
transforms.RandomAffine(
    degrees=0,                    # No rotation (separate from RandomRotation)
    translate=(0.2, 0.2),         # Shift up to 20% horizontally/vertically
    scale=(0.8, 1.2),             # Zoom in/out by up to 20%
    shear=20                      # Shear up to 20 degrees
)
```

### 17.4 How Each Transform Works

| Transform | What It Does | Mathematical Effect |
|-----------|-------------|-------------------|
| **HorizontalFlip** | Mirror image left-right | pixel(x,y) → pixel(W-x, y) |
| **Rotation** | Rotate by random angle θ | Applies 2D rotation matrix to coordinates |
| **RandomResizedCrop** | Crop a random region and resize to target | Changes which part of the image the model sees |
| **Translate** | Shift pixels horizontally/vertically | pixel(x,y) → pixel(x+dx, y+dy) |
| **Scale** | Zoom in or out | Multiplies coordinates by scale factor |
| **Shear** | Tilts image into a parallelogram | Applies shear matrix to coordinates |

### 17.5 Critical Rule

**Only augment the training set. Never augment validation or test sets.**

Validation/test transforms should only resize and normalize:
```python
val_transforms = transforms.Compose([
    transforms.Resize(150),
    transforms.CenterCrop(150),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
```

### 17.6 Trade-offs

- Training accuracy may **decrease** (the augmented data is harder)
- Validation accuracy should **increase over time** (better generalization)
- Training takes **longer** (more computation per image)
- Some augmentations can hurt: random cropping CGI images with centered subjects may cut off important features

---

## 18. Transfer Learning — Standing on the Shoulders of Giants

### 18.1 The Concept

Instead of training a CNN from scratch, **reuse** the convolutional layers from a model that was already trained on a massive dataset (millions of images, thousands of classes). Those layers already know how to detect edges, textures, shapes, and complex features.

```
PRETRAINED MODEL (e.g., Inception V3):
┌────────────────────────────────────────┬─────────────┐
│  Convolutional Layers (FROZEN)         │  FC Layers  │
│  Already know edges, textures, shapes  │  (removed)  │
└────────────────────────────────────────┴─────────────┘
                    ↓
YOUR MODEL:
┌────────────────────────────────────────┬─────────────────┐
│  Convolutional Layers (FROZEN)         │  YOUR FC Layers  │
│  Reused from pretrained model          │  (trained fresh) │
└────────────────────────────────────────┴─────────────────┘
```

### 18.2 Why It Works

**Lower layers** (close to input) learn universal features:
- Layer 1-2: edges, gradients, colors
- Layer 3-5: textures, patterns, simple shapes
- Layer 6+: complex shapes, parts of objects

These features are **universal** — edges look the same whether you're classifying cats, cars, or clothing. Only the final classification layers need to be task-specific.

### 18.3 Implementation with Inception V3

**Step 1: Load the pretrained model**
```python
pre_trained_model = models.inception_v3(pretrained=True, aux_logits=True)
```

**Step 2: Freeze all layers** (prevent their weights from being updated during training)
```python
for name, parameter in pre_trained_model.named_parameters():
    parameter.requires_grad = False
    if 'Mixed_7c' in name:
        break
```

`requires_grad = False` means: don't compute gradients for this parameter → don't update it during training.

**Step 3: Replace the final classification layer**
```python
num_ftrs = pre_trained_model.fc.in_features   # Number of features coming in
pre_trained_model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 1024),     # New hidden layer
    nn.ReLU(),
    nn.Linear(1024, 2)             # 2 outputs for Horse/Human
)
```

Only the new `fc` layers have `requires_grad = True` → only these layers train.

### 18.4 Results

| Method | Epochs | Training Acc | Validation Acc |
|--------|--------|-------------|----------------|
| CNN from scratch | 15 | 95%+ | ~88% |
| Transfer learning (Inception V3) | 3 | 99%+ | 95%+ |

Transfer learning achieves **higher accuracy** with **fewer epochs** because the pretrained filters already detect useful features. The model only needs to learn how to map those features to the two classes.

### 18.5 How Inception V3 Compresses Images

By the time an image passes through Inception's convolutional layers up to `Mixed_7c`, a 299×299 input has been reduced to 8×8 feature maps. These tiny, information-dense maps capture high-level features that are easy for the final dense layers to classify.

---

## 19. Multiclass Classification — Beyond Binary

### 19.1 What Changes

| Component | Binary (2 classes) | Multiclass (N classes) |
|-----------|-------------------|----------------------|
| Output neurons | 1 (sigmoid) or 2 (softmax) | N |
| Activation | Sigmoid or Softmax | Softmax (or none with CrossEntropyLoss) |
| Loss function | BCELoss | CrossEntropyLoss |
| Prediction | threshold at 0.5 | argmax |

### 19.2 Example: Rock, Paper, Scissors (3 classes)

```python
# Replace Inception's fc layer for 3 classes
num_ftrs = pre_trained_model.fc.in_features
pre_trained_model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 1024),
    nn.ReLU(),
    nn.Linear(1024, 3)      # 3 outputs: Paper(0), Rock(1), Scissors(2)
)

criterion = nn.CrossEntropyLoss()   # Handles multi-class
```

**ImageFolder loads classes alphabetically**, so the class ordering is:
- 0 = Paper
- 1 = Rock
- 2 = Scissors

### 19.3 Interpreting Outputs

```
Output: [-2.558, -1.736, 3.847]
         Paper    Rock    Scissors
```

The largest value (3.847) corresponds to class 2 = Scissors. ✓

`torch.max(output, 1)` returns the value **and** the index of the maximum.

---

## 20. Dropout Regularization — Preventing Co-Adaptation

### 20.1 The Problem

During training, neighboring neurons can develop **similar** weights and biases. They become redundant — all detecting the same feature. This wastes capacity and promotes overfitting because the network becomes overspecialized.

### 20.2 The Solution

**Dropout** randomly sets a fraction `p` of neuron outputs to zero during each training step.

```
Without dropout:                    With dropout (p=0.5):
○──○──○──○──○                      ○──○──╳──○──╳
○──○──○──○──○        ──────>       ○──╳──○──○──○
○──○──○──○──○                      ╳──○──○──╳──○
                                    (╳ = dropped out)
```

### 20.3 How It Works Mathematically

During **training**, for each forward pass, independently for each neuron:

```
With probability p:    output = 0           (neuron is dropped)
With probability 1-p:  output = output / (1-p)  (scale up surviving neurons)
```

The scaling by `1/(1-p)` is critical. If we drop 50% of neurons, the total signal would be halved. Scaling by 2 compensates, so the expected output magnitude stays the same.

During **evaluation** (`model.eval()`), **dropout is disabled** — all neurons are active.

### 20.4 In PyTorch

```python
nn.Dropout(0.5)    # Drop 50% of neurons randomly
nn.Dropout(0.25)   # Drop 25% of neurons randomly
```

### 20.5 Where to Place Dropout

Typically between fully-connected layers:

```python
pre_trained_model.fc = nn.Sequential(
    nn.Dropout(0.5),                # Before first FC layer
    nn.Linear(num_ftrs, 1024),
    nn.ReLU(),
    nn.Dropout(0.5),                # After activation, before next FC layer
    nn.Linear(1024, 3)
)
```

### 20.6 Choosing the Dropout Rate

| Rate | Effect |
|------|--------|
| 0.1-0.2 | Mild regularization |
| 0.3-0.5 | Standard / recommended starting point |
| 0.5-0.8 | Aggressive — may undershoot accuracy but prevents heavy overfitting |

> **Rule of thumb**: Start with 0.5 for fully-connected layers and experiment.

### 20.7 Key Properties

1. **Only active during training** — `model.eval()` disables it
2. **No learnable parameters** — it's a fixed stochastic operation
3. **Different neurons are dropped each batch** — the network can't rely on any single neuron
4. **Forces redundancy** — each neuron must learn useful features independently
5. **Acts like training an ensemble** — each dropout pattern creates a slightly different sub-network

---

## 21. Complete Glossary

| Term | Definition |
|------|-----------|
| **Computer Vision** | Field of AI enabling machines to understand visual data |
| **Fashion MNIST** | 70K grayscale images of 10 clothing types, 28×28 pixels each |
| **Dense / Fully-Connected Layer** | A layer where every input connects to every neuron (`nn.Linear`) |
| **Hidden Layer** | Any layer between the input and output layers |
| **Flatten** | Converting a multi-dimensional tensor to 1D (`nn.Flatten`) |
| **ReLU** | `max(0, x)` — most common hidden-layer activation function |
| **Softmax** | Converts raw scores to probabilities summing to 1 |
| **LogSoftmax** | `log(Softmax(x))` — numerically stable log-probabilities |
| **Sigmoid** | `1/(1+e^{-x})` — squashes to (0,1) for binary classification |
| **NLLLoss** | Negative Log-Likelihood Loss — pairs with LogSoftmax |
| **CrossEntropyLoss** | Combines LogSoftmax + NLLLoss in one step |
| **BCELoss** | Binary Cross-Entropy — for single-neuron binary classification |
| **Adam** | Adaptive optimizer combining momentum and per-parameter learning rates |
| **Hyperparameter** | A value you choose before training (learning rate, neuron count, epochs) |
| **Parameter** | A value learned during training (weights, biases) |
| **Overfitting** | Model memorizes training data but fails on unseen data |
| **Early Stopping** | Halting training when a target metric (e.g., accuracy) is reached |
| **Convolution** | Sliding a learned filter over an image to extract features |
| **Filter / Kernel** | Small matrix of learnable weights applied during convolution |
| **Feature Map** | Output of a convolution — highlights a particular feature |
| **Padding** | Adding zeros around image borders to control output size |
| **Stride** | Number of pixels the filter moves at each step |
| **Pooling (Max)** | Taking the maximum value in each region to reduce spatial size |
| **CNN** | Convolutional Neural Network — uses convolutions for feature extraction |
| **DNN** | Deep Neural Network — network with only dense (fully-connected) layers |
| **Image Augmentation** | Random transforms (flip, rotate, zoom) to expand training data |
| **Transfer Learning** | Reusing pretrained convolutional layers in a new model |
| **Inception V3** | Google's pretrained CNN (1000 classes, trained on ImageNet) |
| **ImageFolder** | PyTorch dataset class that loads images from subdirectories as labels |
| **DataLoader** | PyTorch utility for batching, shuffling, and parallel-loading data |
| **model.train()** | Sets model to training mode (dropout active, batch norm uses batch stats) |
| **model.eval()** | Sets model to evaluation mode (dropout off, batch norm uses running stats) |
| **Dropout** | Randomly zeroing neuron outputs during training to prevent overfitting |
| **Logits** | Raw output values from the model (before or after final activation) |
| **argmax** | Index of the maximum value in a tensor — gives predicted class |
| **Ground Truth** | The correct labels in the dataset |
| **Batch** | A subset of training data processed together (e.g., 64 images) |
| **Epoch** | One complete pass through all batches in the training set |
| **Validation Set** | Data used to monitor performance during training (not for weight updates) |
| **Test Set** | Data used for final, unbiased evaluation after all training is complete |

---

## 22. Summary

### Dense Network (Chapter 2)
1. Computer vision teaches machines to recognize image content
2. Fashion MNIST: 70K grayscale 28×28 images across 10 clothing classes
3. A dense network flattens 784 pixels → 128 hidden neurons → 10 output neurons
4. LogSoftmax + NLLLoss measures classification error
5. Adam optimizer adapts learning rates per parameter
6. Achieves ~87% test accuracy after 5 epochs; overfits with more training
7. Early stopping halts training at a target accuracy to prevent overfitting

### Convolutional Neural Network (Chapter 3)
8. Convolutions slide learned filters over images to detect features (edges, textures, shapes)
9. Convolution math: element-wise multiply filter × image region, then sum
10. Padding preserves spatial dimensions; stride controls output size
11. Pooling (max pooling) halves spatial dimensions while keeping strongest activations
12. CNN achieves 91% on Fashion MNIST (vs 89% DNN)
13. Horses/Humans CNN: 3 conv layers, 2 FC layers, sigmoid output, BCELoss — 10.6M parameters
14. Image augmentation (flip, rotate, crop, shear) virtually expands small datasets
15. Transfer learning reuses pretrained convolutional features (Inception V3) — achieves 95%+ validation with only 3 epochs
16. Multiclass classification: change output neurons + loss function to CrossEntropyLoss
17. Dropout randomly disables neurons during training to reduce overfitting

> **The architectural pattern**: `[Conv → ReLU → Pool] × N → Flatten → Dense → Output` is the foundation of modern computer vision. Everything from simple classifiers to GPT's vision encoders builds on this pattern.

---

## 23. Review Questions

1. **Why can't a dense network understand spatial relationships in images? What does flattening lose?**

2. **Fashion MNIST has 28×28 images and 10 classes. How many parameters does a network with architecture `Linear(784,128) → ReLU → Linear(128,10)` have? Show your calculation.**

3. **Write out the Softmax formula. Compute Softmax([3.0, 1.0, -1.0]) by hand.**

4. **What is the mathematical relationship between NLLLoss and CrossEntropyLoss? When would you choose one over the other?**

5. **Explain why BCELoss = -[y·log(ŷ) + (1-y)·log(1-ŷ)] penalizes wrong predictions heavily. What happens when y=1 and ŷ=0.01?**

6. **A 32×32 image passes through Conv2d(3, 16, kernel_size=5, padding=0, stride=1). What is the output shape? Show the formula.**

7. **Compute the number of parameters in Conv2d(16, 32, kernel_size=3). Show your work.**

8. **Why does max pooling have zero learnable parameters?**

9. **Explain the difference between training, validation, and test sets. Why do we need all three?**

10. **A model gets 99% training accuracy but 75% validation accuracy. Diagnose the problem and list 3 techniques to fix it.**

11. **In transfer learning, why do we freeze the pretrained convolutional layers? What would happen if we didn't?**

12. **Why must we scale surviving neuron outputs by 1/(1-p) during dropout? What would happen without scaling?**

13. **Your model has 10M parameters but only 1,000 training images. Is this a problem? Why? What would you do?**

14. **Derive the output shape at every layer for this architecture on a (3, 64, 64) input**:
    ```
    Conv2d(3, 32, kernel_size=3, padding=1) → ReLU → MaxPool(2,2) →
    Conv2d(32, 64, kernel_size=3, padding=1) → ReLU → MaxPool(2,2) →
    Flatten → Linear(?, 256) → Linear(256, 5)
    ```
    **What value replaces `?` in the Linear layer?**

---

> **Phase 2 Complete.** Say **"continue"** to proceed to **Phase 3: Working with Public Datasets** (Chapter 4).
