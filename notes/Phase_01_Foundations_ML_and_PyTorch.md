# Phase 1: Foundations — What is Machine Learning & Getting Started with PyTorch

> **Covers**: Chapter 1 of *AI and ML for Coders in PyTorch* by Laurence Moroney
> **Goal**: Understand the ML paradigm shift, install PyTorch, build and train your first neural network, and grasp the underlying mathematics.

---

## Table of Contents

1. [The Big Idea: What is Machine Learning?](#1-the-big-idea-what-is-machine-learning)
2. [Traditional Programming vs Machine Learning](#2-traditional-programming-vs-machine-learning)
3. [Why Traditional Programming Hits a Wall](#3-why-traditional-programming-hits-a-wall)
4. [The ML Paradigm: Flipping the Axes](#4-the-ml-paradigm-flipping-the-axes)
5. [What is PyTorch?](#5-what-is-pytorch)
6. [Setting Up Your Environment](#6-setting-up-your-environment)
7. [Your First Neural Network — The "Hello World" of ML](#7-your-first-neural-network--the-hello-world-of-ml)
8. [The Mathematics Behind It All](#8-the-mathematics-behind-it-all)
9. [The Training Loop — Line by Line](#9-the-training-loop--line-by-line)
10. [Making Predictions and Inspecting the Network](#10-making-predictions-and-inspecting-the-network)
11. [Key Terminology Glossary](#11-key-terminology-glossary)
12. [Summary](#12-summary)
13. [Review Questions](#13-review-questions)

---

## 1. The Big Idea: What is Machine Learning?

Machine Learning is **not** a new kind of programming language. It is a completely different **paradigm** — a different way of thinking about what a program does.

In one sentence:

> **Machine Learning is a technique where a computer looks at data and its corresponding answers, and figures out the rules that connect them — instead of a human writing those rules by hand.**

That single idea is the foundation of everything in this book. Every chapter, from computer vision to generative AI, is just a more sophisticated version of this concept.

---

## 2. Traditional Programming vs Machine Learning

### Traditional Programming

In traditional programming, **you** (the programmer) figure out the rules. You write them in code. The code acts on data and produces answers.

```
┌──────────┐     ┌──────────┐     ┌──────────┐
│  RULES   │ ──> │  PROGRAM │ ──> │ ANSWERS  │
│ (code)   │     │          │     │          │
└──────────┘     └──────────┘     └──────────┘
                      ▲
                      │
                 ┌──────────┐
                 │   DATA   │
                 └──────────┘
```

**Example 1 — Breakout Game**:
- **Data**: Ball position (x, y), velocity (dx, dy), brick positions
- **Rules** (written by you): "If ball hits brick, remove brick, reverse dy, increase speed"
- **Answer**: Updated game state

**Example 2 — Price-to-Earnings Ratio**:
- **Data**: Stock price = $100, Earnings per share = $5
- **Rules** (written by you): `P/E = price / earnings`
- **Answer**: P/E = 20

In both cases, **you wrote the rules**. The computer just executed them.

### The Pattern

```
Rules + Data ──> Answers
```

This model has been the backbone of software development since its invention. It works beautifully — **as long as you can figure out the rules**.

---

## 3. Why Traditional Programming Hits a Wall

Some problems are **impossible to solve by writing rules**, not because the rules don't exist, but because they are too complex for a human to articulate.

### The Activity Detection Example

Imagine you're building a fitness app that detects what activity a person is doing.

**Walking**: If speed < 4 mph, probably walking. Easy rule. ✅

**Running**: If speed >= 4 mph and < 12 mph, probably running. Still manageable. ✅

**Biking**: If speed >= 12 mph, probably biking. Getting rough, but okay. ✅

**Golfing**: ❌ **Now we're stuck.**

A golfer walks for a bit, stops, swings, walks again, stops again. Their speed pattern looks a lot like "walking + standing still." There's no simple speed-based rule that distinguishes golfing from someone walking around a park with occasional stops.

The data exists (heart rate, GPS, accelerometer, gyroscope readings), and it *does* look different for golfing vs. walking — but writing explicit `if-else` rules to capture all of it is **unfeasible**. There are too many variables, too many edge cases, and the patterns are too subtle.

> **Key insight**: Traditional programming fails when the rules are too complex for humans to express in code.

---

## 4. The ML Paradigm: Flipping the Axes

Here is the breakthrough. Look at traditional programming again:

```
Rules + Data ──> Answers
```

Now **flip it**:

```
Data + Answers ──> Rules
```

Instead of **you** writing the rules, you provide:
- **Data**: The raw sensor readings, images, text, numbers — whatever you have
- **Answers (Labels)**: What those data points correspond to ("this is walking," "this is golfing")

And the computer **figures out the rules** by itself.

```
┌──────────┐     ┌──────────────┐     ┌──────────┐
│   DATA   │ ──> │   LEARNING   │ ──> │  RULES   │
│          │     │  ALGORITHM   │     │ (model)  │
└──────────┘     └──────────────┘     └──────────┘
                      ▲
                      │
                 ┌──────────┐
                 │ ANSWERS  │
                 │ (labels) │
                 └──────────┘
```

### How This Solves the Golfing Problem

1. Collect sensor data from many people doing many activities
2. **Label** each data sample: "This is walking," "This is golfing," "This is running"
3. Feed data + labels into a learning algorithm
4. The algorithm discovers patterns that distinguish golf from walking — patterns you could never have hand-coded

> **This is Machine Learning.** Instead of programming rules, you program the ability to learn rules from examples.

### The Broader View

- **Artificial Intelligence (AI)**: The broad field of making computers think and act like humans
- **Machine Learning (ML)**: A subset of AI — learning rules from data instead of hand-coding them
- **Deep Learning**: A subset of ML — using neural networks with many layers to learn increasingly abstract patterns

```
┌─────────────────────────────────────┐
│         Artificial Intelligence      │
│  ┌───────────────────────────────┐  │
│  │      Machine Learning          │  │
│  │  ┌─────────────────────────┐  │  │
│  │  │     Deep Learning        │  │  │
│  │  └─────────────────────────┘  │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
```

---

## 5. What is PyTorch?

### History

| Year | Event |
|------|-------|
| ~2002 | **Torch** is created — an open-source ML framework written in Lua |
| 2017 | **PyTorch** is released by Meta (Facebook) AI — a Python port of Torch |
| 2022 | PyTorch moves to the **Linux Foundation** (now independent of Meta) |
| 2023+ | Generative AI explosion makes PyTorch the dominant ML framework |

### Why "torch"?

When you `pip install` PyTorch, the package is called `torch` — a direct reference to its ancestor. This is why you write `import torch` in Python.

### The PyTorch Ecosystem

PyTorch is not just one library. It's an **ecosystem** of tools:

| Component | What It Does |
|-----------|-------------|
| **torch** | Core tensor library + autograd (automatic differentiation) |
| **torch.nn** | Neural network layers, loss functions, containers |
| **torch.optim** | Optimizers (SGD, Adam, etc.) |
| **torchvision** | Datasets, pretrained models, and transforms for computer vision |
| **torchtext** | Tools for NLP tasks |
| **torchaudio** | Tools for audio processing |
| **TorchServe** | Deploy models at scale via REST APIs |
| **PyTorch Mobile** | Run models on Android and iOS |
| **torch.distributed** | Train across multiple GPUs/machines |

### Three Ways to Use Models

1. **Create from scratch** — design and train your own architecture
2. **Use someone else's model** — download a pretrained model and use it directly
3. **Transfer learning** — take a pretrained model, freeze most of it, and retrain the last layers on your data

### Training vs Inference

| Term | Definition |
|------|-----------|
| **Training** | The process of teaching a model by showing it data + labels repeatedly |
| **Inference** | Using a trained model to make predictions on new, unseen data |

### Hardware

| Device | Use Case |
|--------|---------|
| **CPU** | Fine for simple models, always available |
| **GPU** (NVIDIA CUDA) | Much faster for training — parallelizes matrix math |
| **TPU** (Google) | Specialized chip for tensor operations, available in Colab |
| **Metal** (Apple) | GPU acceleration on Mac |

---

## 6. Setting Up Your Environment

### Option A: pip install (simplest)

```bash
pip install torch
```

Verify:

```python
import torch
print(torch.__version__)   # e.g., "2.4.1"
```

### Option B: PyCharm (recommended for debugging)

1. Create a new project with a **Conda virtual environment**
2. Go to `File → Settings → Project → Python Interpreter`
3. Click `+`, search for `torch`, click **Install Package**

**Why virtual environments matter**: You can have PyTorch 1.x in one project and 2.x in another without conflicts.

### Option C: Google Colab (easiest, free GPU)

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **+ New notebook**
3. PyTorch comes pre-installed. Verify:

```python
import torch
print(torch.__version__)   # e.g., "2.4.1+cu121"
```

The `+cu121` means CUDA 12.1 drivers are installed — you get GPU acceleration for free.

> **Note**: Colab's PyTorch version may lag behind the latest release. You can upgrade with:
> ```python
> !pip install torch==<version>
> ```
> But be careful — upgrading may lose CUDA drivers, falling back to CPU.

---

## 7. Your First Neural Network — The "Hello World" of ML

### The Problem

Given these (x, y) pairs:

| x  | y  |
|----|----|
| -1 | -3 |
|  0 | -1 |
|  1 |  1 |
|  2 |  3 |
|  3 |  5 |
|  4 |  7 |

**Task**: Can the computer figure out the relationship between x and y?

**Human reasoning**: x goes up by 1, y goes up by 2. When x = 0, y = -1. So y = 2x - 1.

**ML approach**: Don't figure out the formula yourself. Give the computer the data and let it learn.

### The Complete Code

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Define the model
model = nn.Sequential(nn.Linear(1, 1))

# 2. Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 3. Prepare the data
xs = torch.tensor([[-1.0], [0.0], [1.0], [2.0], [3.0], [4.0]],
                   dtype=torch.float32)
ys = torch.tensor([[-3.0], [-1.0], [1.0], [3.0], [5.0], [7.0]],
                   dtype=torch.float32)

# 4. Train
for epoch in range(500):
    optimizer.zero_grad()       # Reset gradients
    outputs = model(xs)         # Forward pass
    loss = criterion(outputs, ys)  # Compute loss
    loss.backward()             # Backward pass (compute gradients)
    optimizer.step()            # Update weights

# 5. Predict
with torch.no_grad():
    prediction = model(torch.tensor([[10.0]], dtype=torch.float32))
    print(prediction)  # Should be close to 19.0
```

### Line-by-Line Walkthrough

Let's break every single line down.

---

#### Line: `import torch`

Imports the core PyTorch library. This gives you access to tensors (the fundamental data structure) and the autograd engine (automatic differentiation).

#### Line: `import torch.nn as nn`

Imports the **neural network** module. Contains:
- Layer types (`nn.Linear`, `nn.Conv2d`, etc.)
- Loss functions (`nn.MSELoss`, `nn.CrossEntropyLoss`, etc.)
- Containers (`nn.Sequential`, `nn.Module`)

#### Line: `import torch.optim as optim`

Imports **optimizers** — algorithms that update the model's parameters to reduce loss.

---

#### Line: `model = nn.Sequential(nn.Linear(1, 1))`

This is the most important line. Let's dissect it piece by piece.

**`nn.Sequential(...)`**: A container that chains layers in order. Data flows through layer 1, then layer 2, then layer 3, etc. Here, we only have one layer.

**`nn.Linear(1, 1)`**: A **linear layer** (also called a **fully connected** or **dense** layer).

What does `nn.Linear(in_features, out_features)` do? It computes:

```
output = input × weight + bias
```

Or in math notation:

$$y = xW^T + b$$

The parameters `(1, 1)` mean:
- `in_features = 1` → the layer takes **1 input value**
- `out_features = 1` → the layer produces **1 output value**

So this layer has:
- **1 weight** (a single number, `W`)
- **1 bias** (a single number, `b`)

Total learnable parameters: **2** (one weight + one bias)

**The neural network looks like this**:

```
    ┌─────────────┐
x ──│  W, b       │── y
    │  y = Wx + b │
    └─────────────┘
     One neuron
     One layer
```

This is the simplest possible neural network: one layer, one neuron.

---

#### Line: `criterion = nn.MSELoss()`

This defines the **loss function** — the way we measure "how wrong" the model's predictions are. MSE stands for **Mean Squared Error**. Full math in Section 8.

---

#### Line: `optimizer = optim.SGD(model.parameters(), lr=0.01)`

This defines the **optimizer** — the algorithm that adjusts the model's weights and bias to reduce the loss.

- **`optim.SGD`**: Stochastic Gradient Descent (full math in Section 8)
- **`model.parameters()`**: Tells the optimizer which values it's allowed to change (W and b)
- **`lr=0.01`**: The **learning rate** — how big each adjustment step is

---

#### Lines: Preparing the data

```python
xs = torch.tensor([[-1.0], [0.0], [1.0], [2.0], [3.0], [4.0]],
                   dtype=torch.float32)
ys = torch.tensor([[-3.0], [-1.0], [1.0], [3.0], [5.0], [7.0]],
                   dtype=torch.float32)
```

**`torch.tensor`**: Creates a **tensor** — PyTorch's fundamental data structure.

**Why the double brackets `[[-1.0], [0.0], ...]`?**

`nn.Linear(1, 1)` expects input of shape `(batch_size, in_features)`. Each sample is a row, and each row has 1 feature. So:

```
xs shape: (6, 1)  →  6 samples, each with 1 feature
ys shape: (6, 1)  →  6 samples, each with 1 label
```

**`dtype=torch.float32`**: Neural networks work with 32-bit floating point numbers. This is standard practice — integer types would not work with gradient computation.

---

#### The Training Loop

```python
for epoch in range(500):
    optimizer.zero_grad()
    outputs = model(xs)
    loss = criterion(outputs, ys)
    loss.backward()
    optimizer.step()
```

This loop runs **500 times**. Each full pass through the loop is called an **epoch**. Here's what each line does, and the diagrams follow in the math section:

| Line | What It Does | Why |
|------|-------------|-----|
| `optimizer.zero_grad()` | Sets all gradients to zero | Gradients accumulate by default; we need a fresh start each epoch |
| `outputs = model(xs)` | **Forward pass** — compute predictions for all 6 inputs | This is the "guess" step |
| `loss = criterion(outputs, ys)` | Compare predictions to correct answers | Measures how bad the guess is |
| `loss.backward()` | **Backward pass** — compute gradients | Calculates which direction to adjust W and b |
| `optimizer.step()` | Update W and b using the gradients | Makes the model slightly better |

---

#### Line: Making a prediction

```python
with torch.no_grad():
    prediction = model(torch.tensor([[10.0]], dtype=torch.float32))
    print(prediction)
```

- **`torch.no_grad()`**: Disables gradient tracking. We're not training — just predicting. This saves memory and computation.
- We feed in `x = 10.0` and get back a `y` value close to 19.0 (since y = 2(10) - 1 = 19).
- The answer won't be **exactly** 19 — it will be something like 18.991. This is because the model learned `W ≈ 1.9987` and `b ≈ -0.9960`, not exactly `W = 2, b = -1`.

---

## 8. The Mathematics Behind It All

This is the section where we go deep. Every equation is derived from scratch. No hand-waving.

### 8.1 Tensors

A **tensor** is a generalization of familiar data structures:

| Dimensions | Mathematical Name | Python Analogy | Example |
|-----------|------------------|---------------|---------|
| 0 | Scalar | `x = 5` | A single number |
| 1 | Vector | `[1, 2, 3]` | A list of numbers |
| 2 | Matrix | `[[1,2],[3,4]]` | A table of numbers |
| 3 | 3D Tensor | `[[[1,2],[3,4]],[[5,6],[7,8]]]` | A cube of numbers |
| n | nD Tensor | Nested arrays | Higher-dimensional data |

In PyTorch:
```python
scalar = torch.tensor(5)              # shape: ()
vector = torch.tensor([1, 2, 3])      # shape: (3,)
matrix = torch.tensor([[1,2],[3,4]])   # shape: (2, 2)
```

**Why tensors?** Neural networks process data as multi-dimensional arrays. A batch of 32 color images of size 224x224 is a tensor of shape `(32, 3, 224, 224)` — 32 images, 3 color channels (RGB), 224 pixels tall, 224 pixels wide.

---

### 8.2 The Linear Model (What One Neuron Computes)

A single neuron with one input computes:

```
y = Wx + b
```

Where:
- **x** = input value (given)
- **W** = weight (learned)
- **b** = bias (learned)
- **y** = output (predicted)

This is the equation of a straight line. The neuron's job is to find the **W** and **b** that make the line fit the data.

**In our example**: The true relationship is `y = 2x - 1`, so the ideal learned values are `W = 2` and `b = -1`.

**Generalization to multiple inputs**: If we have `n` inputs instead of 1:

```
y = W₁x₁ + W₂x₂ + ... + Wₙxₙ + b
```

Or in vector notation:

```
y = W · x + b
```

Where **W** is a vector of weights and **x** is a vector of inputs. The dot product (`W · x`) is the sum of element-wise products.

---

### 8.3 The Loss Function: Mean Squared Error (MSE)

The loss function answers: **"How wrong is the model right now?"**

**Mean Squared Error (MSE)** is defined as:

```
              1   n
MSE = L(W,b) = ─── Σ (yᵢ - ŷᵢ)²
              n  i=1
```

Where:
- **n** = number of data points (in our case, 6)
- **yᵢ** = the true/correct value for data point i
- **ŷᵢ** (y-hat) = the model's predicted value for data point i
- **(yᵢ - ŷᵢ)** = the error (also called **residual**) for data point i
- **(yᵢ - ŷᵢ)²** = the squared error — always positive, penalizes large errors more

#### Worked Example

Suppose the model's current guess is `y = 10x + 10` (W=10, b=10). Let's compute MSE:

| i | xᵢ  | yᵢ (true) | ŷᵢ = 10xᵢ+10 | Error (yᵢ - ŷᵢ) | Squared Error |
|---|------|-----------|---------------|------------------|---------------|
| 1 | -1   | -3        | 0             | -3               | 9             |
| 2 |  0   | -1        | 10            | -11              | 121           |
| 3 |  1   |  1        | 20            | -19              | 361           |
| 4 |  2   |  3        | 30            | -27              | 729           |
| 5 |  3   |  5        | 40            | -35              | 1225          |
| 6 |  4   |  7        | 50            | -43              | 1849          |

```
MSE = (9 + 121 + 361 + 729 + 1225 + 1849) / 6
    = 4294 / 6
    = 715.67
```

That's a **huge** loss — the guess `y = 10x + 10` is terrible.

Now suppose the model has learned `y = 1.999x - 0.996`. Same calculation:

| i | xᵢ  | yᵢ (true) | ŷᵢ ≈ 1.999xᵢ - 0.996 | Error | Squared Error |
|---|------|-----------|----------------------|-------|---------------|
| 1 | -1   | -3        | -2.995               | -0.005 | 0.000025     |
| 2 |  0   | -1        | -0.996               | -0.004 | 0.000016     |
| 3 |  1   |  1        |  1.003               | -0.003 | 0.000009     |
| 4 |  2   |  3        |  3.002               | -0.002 | 0.000004     |
| 5 |  3   |  5        |  5.001               | -0.001 | 0.000001     |
| 6 |  4   |  7        |  7.000               |  0.000 | 0.000000     |

```
MSE ≈ 0.000055 / 6 ≈ 0.0000092
```

The loss is near zero — the model has essentially learned the correct relationship.

#### Why Squared? Why Not Just Absolute Value?

1. **Squaring makes all errors positive** — a prediction that's too high and one that's too low don't cancel each other out
2. **Squaring penalizes large errors more** — an error of 10 contributes 100, while an error of 1 contributes only 1. This pushes the optimizer to fix the worst predictions first.
3. **The squared function is differentiable everywhere** — the absolute value function has a kink at zero, which makes calculus harder

---

### 8.4 Gradient Descent: How the Model Learns

The goal of training is to find the W and b that **minimize the loss function**. Gradient descent is the algorithm that does this.

#### The Core Idea

Imagine you're standing on a hilly landscape in thick fog. You can't see the lowest point, but you can **feel the slope under your feet**. To get to the lowest point, you take a step in the direction the ground slopes downward. You repeat this until the ground feels flat (you've reached the minimum).

```
Loss
 │
 │   X ← You start here (random W, b)
 │  / \
 │ /   \
 │/     \
 │       \     ← You follow the slope down
 │        \
 │         ·  ← Minimum loss (best W, b)
 └──────────────── W
```

The "slope" is the **gradient** — the derivative of the loss with respect to the parameters.

#### The Math

We want to minimize `L(W, b) = MSE`. To do this, we need to know:
- **How does the loss change when we change W?** → This is `∂L/∂W` (partial derivative of L with respect to W)
- **How does the loss change when we change b?** → This is `∂L/∂b` (partial derivative of L with respect to b)

Let's derive them.

Our model predicts: `ŷᵢ = Wxᵢ + b`

Our loss is:

```
         1   n
L(W,b) = ─── Σ (yᵢ - ŷᵢ)²
         n  i=1

         1   n
       = ─── Σ (yᵢ - Wxᵢ - b)²
         n  i=1
```

**Derivative with respect to W** (using the chain rule):

```
∂L      1   n                         
── = - ─── Σ  2(yᵢ - Wxᵢ - b) · xᵢ  
∂W     n  i=1                         

       -2   n
     = ─── Σ (yᵢ - Wxᵢ - b) · xᵢ
        n  i=1
```

Step by step:
1. The outer function is `(something)²` → derivative is `2 · (something)`
2. The inner function is `(yᵢ - Wxᵢ - b)` → derivative with respect to W is `-xᵢ`
3. Chain rule: multiply them → `2(yᵢ - Wxᵢ - b) · (-xᵢ)`
4. Average over all n data points

**Derivative with respect to b**:

```
∂L      -2   n
── =   ─── Σ (yᵢ - Wxᵢ - b)
∂b      n  i=1
```

Same steps, except the inner derivative with respect to b is `-1` instead of `-xᵢ`.

#### The Update Rule

Once we have the gradients, we update:

```
W_new = W_old - α · (∂L/∂W)
b_new = b_old - α · (∂L/∂b)
```

Where **α** (alpha) is the **learning rate** — how big each step is.

- **α too large** → you overshoot the minimum and bounce around
- **α too small** → you creep toward the minimum painfully slowly
- **α just right** → you converge efficiently

```
α too large:          α too small:         α just right:
                                                                    
Loss                  Loss                 Loss                     
 │ X    X             │ X                  │ X                      
 │  \  /              │  \                 │  \                     
 │   \/               │   \               │   \                    
 │   /\               │    \              │    \                   
 │  X  X              │     x             │     ·                  
 │   DIVERGE          │      x x x x     │   CONVERGE             
 └──────── W          └──────── W         └──────── W              
```

#### Why "Stochastic" Gradient Descent (SGD)?

- **Gradient Descent**: Compute gradients using ALL data points, then update once
- **Stochastic Gradient Descent**: Compute gradients using a random SUBSET (or even a single data point), then update
- **Mini-batch SGD** (most common): Compute gradients using a small random batch (e.g., 32 samples)

In our tiny example with only 6 data points, the distinction doesn't matter. In real-world datasets with millions of samples, SGD is critical for practical speed.

In our code, `optim.SGD(model.parameters(), lr=0.01)` means: use SGD with a learning rate of 0.01.

---

### 8.5 Backpropagation: The Chain Rule at Scale

In our simple one-neuron network, computing gradients is straightforward. But in deep networks with many layers, computing `∂Loss/∂(every parameter)` by hand would be impossible.

**Backpropagation** is an algorithm that computes gradients efficiently by applying the **chain rule** of calculus **backwards** through the network.

#### The Chain Rule (Quick Refresher)

If `y = f(g(x))`, then:

```
dy/dx = f'(g(x)) · g'(x)
```

In words: the derivative of a composition of functions is the product of the derivatives of each function.

#### How Backpropagation Works (Conceptual)

Consider a network with 3 layers:

```
x ──[Layer 1]──> h₁ ──[Layer 2]──> h₂ ──[Layer 3]──> ŷ ──[Loss]──> L
```

To find how Layer 1's weights affect the loss:

```
∂L/∂W₁ = (∂L/∂ŷ) · (∂ŷ/∂h₂) · (∂h₂/∂h₁) · (∂h₁/∂W₁)
```

The algorithm works **backwards**:
1. Compute `∂L/∂ŷ` (how loss changes with output)
2. Compute `∂ŷ/∂h₂` (how output changes with layer 3's input)
3. Compute `∂h₂/∂h₁` (how layer 3's input changes with layer 2's input)
4. Compute `∂h₁/∂W₁` (how layer 2's input changes with layer 1's weights)

Each step reuses the results from the previous step (this is the "backward pass"). This is what `loss.backward()` does in PyTorch.

#### In Our One-Neuron Example

Our computation graph is:

```
x ──[W·x + b]──> ŷ ──[MSE]──> L
```

Backward:
1. `∂L/∂ŷ = (2/n) · (ŷ - y)` (derivative of MSE)
2. `∂ŷ/∂W = x` (derivative of `Wx + b` w.r.t. W)
3. `∂ŷ/∂b = 1` (derivative of `Wx + b` w.r.t. b)

Combine:
- `∂L/∂W = ∂L/∂ŷ · ∂ŷ/∂W = (2/n)(ŷ - y) · x`
- `∂L/∂b = ∂L/∂ŷ · ∂ŷ/∂b = (2/n)(ŷ - y) · 1`

This matches what we derived in section 8.4, confirming that backpropagation gives the correct gradients.

> **The beautiful part**: PyTorch does all of this automatically. When you call `loss.backward()`, it walks backward through every operation you performed and computes all gradients. You never have to derive them by hand.

---

### 8.6 Putting It All Together — One Epoch Walkthrough

Let's trace one complete epoch with concrete numbers.

**Setup**: W = 10, b = 10 (random initial guess), learning rate α = 0.01

**Step 1: Forward Pass** (`outputs = model(xs)`)

| xᵢ  | ŷᵢ = 10·xᵢ + 10 | yᵢ (true) |
|------|-------------------|-----------|
| -1   | 0                 | -3        |
|  0   | 10                | -1        |
|  1   | 20                |  1        |
|  2   | 30                |  3        |
|  3   | 40                |  5        |
|  4   | 50                |  7        |

**Step 2: Compute Loss** (`loss = criterion(outputs, ys)`)

```
MSE = [(−3−0)² + (−1−10)² + (1−20)² + (3−30)² + (5−40)² + (7−50)²] / 6
    = [9 + 121 + 361 + 729 + 1225 + 1849] / 6
    = 4294 / 6
    = 715.67
```

**Step 3: Backward Pass** (`loss.backward()`)

Compute gradients:

```
∂L/∂W = (-2/6) · Σ (yᵢ - ŷᵢ) · xᵢ
       = (-2/6) · [(-3)(−1) + (−11)(0) + (−19)(1) + (−27)(2) + (−35)(3) + (−43)(4)]
       = (-2/6) · [3 + 0 + (−19) + (−54) + (−105) + (−172)]
       = (-2/6) · (−347)
       = 115.67

∂L/∂b = (-2/6) · Σ (yᵢ - ŷᵢ)
       = (-2/6) · [(−3) + (−11) + (−19) + (−27) + (−35) + (−43)]
       = (-2/6) · (−138)
       = 46.0
```

**Step 4: Update Parameters** (`optimizer.step()`)

```
W_new = W_old - α · ∂L/∂W = 10 - 0.01 · 115.67 = 10 - 1.1567 = 8.8433
b_new = b_old - α · ∂L/∂b = 10 - 0.01 · 46.0   = 10 - 0.46   = 9.54
```

After just **one epoch**, the model went from `y = 10x + 10` to `y = 8.84x + 9.54`. The loss will drop. After 500 epochs, it converges to `y ≈ 2x - 1`.

---

### 8.7 Why Zero the Gradients?

```python
optimizer.zero_grad()
```

PyTorch **accumulates** gradients by default. If you don't zero them, the gradient from epoch 1 gets added to the gradient from epoch 2, and so on. This is by design (useful in some advanced scenarios), but for standard training, you must reset them at the start of each epoch.

Without zeroing:
```
Epoch 1: gradient = 5
Epoch 2: gradient = 5 + 3 = 8  ← Wrong! Should be 3
Epoch 3: gradient = 8 + 1 = 9  ← Wrong! Should be 1
```

---

## 9. The Training Loop — Line by Line

Here is the training loop one more time, with a visual flow:

```python
for epoch in range(500):        # Repeat 500 times
    optimizer.zero_grad()        # 1. Clear old gradients
    outputs = model(xs)          # 2. Forward pass: compute predictions
    loss = criterion(outputs, ys) # 3. Compute loss
    loss.backward()              # 4. Backward pass: compute gradients
    optimizer.step()             # 5. Update weights
```

Visual flow for **one epoch**:

```
                    ┌────────────────────────────┐
                    │  1. zero_grad()             │
                    │  Reset all gradients to 0   │
                    └─────────────┬──────────────┘
                                  ▼
                    ┌────────────────────────────┐
                    │  2. model(xs)               │
                    │  FORWARD PASS               │
                    │  ŷ = W·x + b for all x     │
                    └─────────────┬──────────────┘
                                  ▼
                    ┌────────────────────────────┐
                    │  3. criterion(outputs, ys)  │
                    │  COMPUTE LOSS               │
                    │  L = MSE(ŷ, y)             │
                    └─────────────┬──────────────┘
                                  ▼
                    ┌────────────────────────────┐
                    │  4. loss.backward()         │
                    │  BACKWARD PASS              │
                    │  Compute ∂L/∂W and ∂L/∂b   │
                    └─────────────┬──────────────┘
                                  ▼
                    ┌────────────────────────────┐
                    │  5. optimizer.step()        │
                    │  UPDATE PARAMETERS          │
                    │  W = W - α·∂L/∂W           │
                    │  b = b - α·∂L/∂b           │
                    └─────────────┬──────────────┘
                                  ▼
                          Next epoch...
```

### What the Loss Looks Like Over Time

```
Epoch     Loss
  1       ~5.64        ← Initial random guess is bad
  5       ~2.10        ← Getting better fast
 10       ~0.86        ← 6x improvement from epoch 1
 50       ~0.01        ← Almost there
100       ~0.0002      ← Very close
500       ~0.0000095   ← Essentially perfect
```

The loss decreases rapidly at first (when the model is far from the answer) and slows down as it approaches the minimum. This is characteristic of gradient descent — large errors produce large gradients, which produce large updates.

---

## 10. Making Predictions and Inspecting the Network

### Prediction

```python
with torch.no_grad():
    prediction = model(torch.tensor([[10.0]], dtype=torch.float32))
    print(prediction)  # Output: tensor([[18.9910]])  (approximately)
```

**Why not exactly 19?**
1. The loss never reached exactly 0 — it reached ~0.0000095
2. The model was trained on only 6 data points
3. W learned ≈ 1.9987, not exactly 2; b learned ≈ -0.9960, not exactly -1

**The answer will differ slightly each time you run it** because the initial random values of W and b are different on each run.

> **Terminology**: We call this a "prediction" not because it's looking into the future, but because there's a degree of uncertainty. The model is predicting based on what it learned — it hasn't seen x=10 before.

### Inspecting the Learned Parameters

```python
layer = model[0]                       # Get the first (only) layer
weights = layer.weight.data.numpy()    # Get W as a NumPy array
bias = layer.bias.data.numpy()         # Get b as a NumPy array

print("Weight:", weights)   # Output: [[1.998695]]
print("Bias:", bias)        # Output: [-0.9959542]
```

**Interpretation**: The model learned `y = 1.9987x - 0.9960`, which is very close to the true relationship `y = 2x - 1`.

The small discrepancy is **expected and healthy** — it means the model generalizes rather than memorizing the exact data.

---

## 11. Key Terminology Glossary

| Term | Definition |
|------|-----------|
| **Artificial Intelligence (AI)** | The broad field of making computers think/act like humans |
| **Machine Learning (ML)** | Learning rules from data + labels, instead of hand-coding rules |
| **Deep Learning** | ML using neural networks with multiple layers |
| **Neural Network** | A model composed of layers of neurons that learn patterns from data |
| **Neuron** | The basic unit — computes `output = activation(Σ(weight × input) + bias)` |
| **Layer** | A collection of neurons that process data together |
| **Weight (W)** | A learnable parameter that scales an input |
| **Bias (b)** | A learnable parameter that shifts the output |
| **Tensor** | A multi-dimensional array — the fundamental data structure in PyTorch |
| **Forward Pass** | Computing predictions by passing data through the network |
| **Loss Function** | A function that measures how wrong the model's predictions are |
| **MSE (Mean Squared Error)** | Average of squared differences between predicted and true values |
| **Gradient** | The derivative of the loss with respect to a parameter — tells the direction of steepest increase |
| **Gradient Descent** | Iteratively moving parameters in the opposite direction of the gradient to minimize loss |
| **SGD (Stochastic Gradient Descent)** | Gradient descent using random subsets of data per step |
| **Learning Rate (α or lr)** | Controls how big each parameter update step is |
| **Backpropagation** | Algorithm that efficiently computes gradients using the chain rule backwards through the network |
| **Epoch** | One complete pass through the entire training dataset |
| **Training** | The process of adjusting model parameters by repeatedly computing loss and updating weights |
| **Inference / Prediction** | Using a trained model on new data to get outputs |
| **Parameters** | The learnable values in a model (weights and biases) |
| **`nn.Sequential`** | A PyTorch container that chains layers in order |
| **`nn.Linear(in, out)`** | A fully connected layer computing `y = xW^T + b` |
| **`loss.backward()`** | Triggers backpropagation — computes all gradients |
| **`optimizer.step()`** | Updates all parameters using the computed gradients |
| **`optimizer.zero_grad()`** | Resets all gradients to zero before the next epoch |
| **`torch.no_grad()`** | Context manager that disables gradient tracking (used during inference) |

---

## 12. Summary

1. **Traditional programming** = you write rules; the computer applies them to data to get answers
2. **Machine learning** = you provide data + answers; the computer learns the rules
3. **PyTorch** is an open-source ML framework (originally from Meta, now Linux Foundation) that makes it easy to build and train neural networks in Python
4. A **neural network** is made of layers of neurons. Each neuron computes `y = Wx + b`
5. **Training** works in a loop:
   - Forward pass → compute predictions
   - Loss function → measure error (MSE)
   - Backward pass → compute gradients via backpropagation
   - Optimizer step → update weights using gradient descent
6. After training, the model can **predict** outputs for inputs it has never seen
7. The same code pattern (define model → define loss → define optimizer → training loop → predict) scales from this trivial example to billion-parameter models

> **The pattern you learned in this chapter is the same pattern used in every chapter of this book.** Only the model architecture, data, and loss function change.

---

## 13. Review Questions

Test your understanding. Try to answer without looking back.

1. **What is the key difference between traditional programming and machine learning?**

2. **In the ML diagram, what are the three components? Which one does the computer figure out?**

3. **What does `nn.Linear(1, 1)` create? How many learnable parameters does it have?**

4. **Write out the MSE formula from memory. Why do we square the errors?**

5. **What is the gradient of the loss with respect to W? (Derive it for the simple case y = Wx + b with MSE loss.)**

6. **What does `optimizer.zero_grad()` do, and why is it necessary?**

7. **What is the learning rate? What happens if it's too high? Too low?**

8. **Why is the prediction for x=10 not exactly 19?**

9. **What is the difference between training and inference?**

10. **In the training loop, what is the order of operations? (List the 5 steps.)**

---

> **Phase 1 Complete.** Say **"continue"** to proceed to **Phase 2: Computer Vision — From Dense Networks to CNNs** (Chapters 2-3).
