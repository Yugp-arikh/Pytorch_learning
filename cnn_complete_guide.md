# Convolutional Neural Networks — The Complete Guide

> **The one goal of a CNN:** Learn spatial features directly from raw pixel data — edges, textures, shapes, objects — by sliding small learnable filters across an image, building up increasingly abstract representations layer by layer.

---

## Table of Contents

1. [Why CNNs? The Problem with Fully Connected Networks](#1-why-cnns)
2. [All Notation Defined](#2-all-notation-defined)
3. [Images as Tensors](#3-images-as-tensors)
4. [The Convolution Operation](#4-the-convolution-operation)
5. [Padding](#5-padding)
6. [Stride](#6-stride)
7. [Multiple Filters — Feature Maps](#7-multiple-filters--feature-maps)
8. [Multi-Channel Convolution (RGB Images)](#8-multi-channel-convolution-rgb-images)
9. [Activation Functions](#9-activation-functions)
10. [Pooling Layers](#10-pooling-layers)
11. [Batch Normalisation](#11-batch-normalisation)
12. [Dropout](#12-dropout)
13. [Fully Connected Layers](#13-fully-connected-layers)
14. [The Softmax + Cross-Entropy Loss](#14-the-softmax--cross-entropy-loss)
15. [The Full Forward Pass](#15-the-full-forward-pass)
16. [Backpropagation Through a CNN](#16-backpropagation-through-a-cnn)
17. [Weight Initialisation](#17-weight-initialisation)
18. [Classic Architectures](#18-classic-architectures)
19. [PyTorch — Complete Implementation](#19-pytorch--complete-implementation)
20. [Training Loop in Full Detail](#20-training-loop-in-full-detail)
21. [Data Augmentation](#21-data-augmentation)
22. [Common Bugs and How to Fix Them](#22-common-bugs-and-how-to-fix-them)
23. [Quick Reference](#23-quick-reference)

---

## 1. Why CNNs?

### The problem with fully connected networks on images

Take a modest $224 \times 224$ RGB image. That's:

$$
224 \times 224 \times 3 = 150{,}528 \text{ input pixels}
$$

If you flatten this and feed it into a fully connected layer with just $1{,}000$ hidden neurons, the weight matrix alone has:

$$
150{,}528 \times 1{,}000 = 150{,}528{,}000 \text{ parameters}
$$

That's **150 million parameters in one layer**. Problems:

1. **Computationally expensive** — too many multiplications per forward pass
2. **Overfits easily** — too many parameters, not enough data
3. **Ignores spatial structure** — a pixel at position $(10, 10)$ has no special relationship to the weight connecting it; the network must re-learn that "pixels near each other are related" from scratch
4. **Not translation invariant** — if the cat moves 2 pixels to the right, the network sees a completely different input

### How CNNs solve this

CNNs exploit three key ideas:

- **Local connectivity:** Each neuron only connects to a small local region (e.g. $3 \times 3$) of the input, not the whole image
- **Weight sharing:** The same filter (set of weights) is slid across the entire image — one filter, millions of positions, same weights
- **Hierarchical features:** Early layers detect edges → middle layers detect textures/shapes → deep layers detect objects

A $3 \times 3$ filter has only $9$ weights (plus 1 bias). That same filter is applied at every spatial position in the image. This reduces parameters from millions to tens.

---

## 2. All Notation Defined

| Symbol | Meaning | Example |
|--------|---------|---------|
| $H$ | Height of input | $28$ (pixels) |
| $W$ | Width of input | $28$ (pixels) |
| $C_{in}$ | Number of input channels | $3$ (RGB) |
| $C_{out}$ | Number of output channels (filters) | $32$ |
| $K$ | Kernel/filter size (assumed square) | $3$ means $3 \times 3$ |
| $P$ | Padding (pixels added to each side) | $1$ |
| $S$ | Stride (step size when sliding filter) | $1$ or $2$ |
| $\mathbf{X}$ | Input tensor | Shape $(C_{in}, H, W)$ |
| $\mathbf{K}$ | Kernel/filter tensor | Shape $(C_{out}, C_{in}, K, K)$ |
| $\mathbf{Z}$ | Pre-activation feature map | Shape $(C_{out}, H_{out}, W_{out})$ |
| $\mathbf{A}$ | Post-activation feature map | Shape $(C_{out}, H_{out}, W_{out})$ |
| $H_{out}$ | Output height | $\lfloor (H + 2P - K) / S \rfloor + 1$ |
| $W_{out}$ | Output width | $\lfloor (W + 2P - K) / S \rfloor + 1$ |
| $b$ | Bias (one per output channel) | Scalar added to whole feature map |
| $*$ | Convolution operator | Slides filter over input |
| $\odot$ | Element-wise (Hadamard) product | Same-shape tensors multiplied entry-by-entry |
| $\sigma$ | Activation function | ReLU, sigmoid, etc. |
| $N$ | Batch size | $32$ images at once |
| $L$ | Loss | Cross-entropy or MSE |
| $\eta$ | Learning rate | $0.001$ |
| $\delta$ | Gradient (error signal) | Same shape as the tensor it refers to |

---

## 3. Images as Tensors

### Grayscale image

A grayscale image is a 2D matrix of pixel intensities:

$$
\mathbf{X} \in \mathbb{R}^{H \times W}
$$

Each entry is a number, typically in $[0, 255]$ (uint8) or $[0.0, 1.0]$ (normalised float).

**Example — a $4 \times 4$ grayscale image:**

$$
\mathbf{X} = \begin{pmatrix}
10 & 20 & 30 & 40 \\
50 & 60 & 70 & 80 \\
90 & 100 & 110 & 120 \\
130 & 140 & 150 & 160
\end{pmatrix}
$$

### RGB image

A colour image has 3 channels — Red, Green, Blue:

$$
\mathbf{X} \in \mathbb{R}^{C_{in} \times H \times W} = \mathbb{R}^{3 \times H \times W}
$$

In PyTorch the convention is **channels first**: $(C, H, W)$.
In NumPy/PIL the convention is **channels last**: $(H, W, C)$.

### Batch of images

In practice you process $N$ images simultaneously (a batch):

$$
\mathbf{X}_{batch} \in \mathbb{R}^{N \times C_{in} \times H \times W}
$$

This is the standard shape of every tensor flowing through a PyTorch CNN.

### Normalisation

Before feeding into a network, pixel values are normalised:

$$
\mathbf{X}_{norm} = \frac{\mathbf{X} - \mu}{\sigma}
$$

where $\mu$ and $\sigma$ are the mean and standard deviation computed over the training set (per channel). For ImageNet:

$$
\mu = [0.485,\ 0.456,\ 0.406], \qquad \sigma = [0.229,\ 0.224,\ 0.225]
$$

---

## 4. The Convolution Operation

### Intuition

Imagine you have a small template (the filter/kernel), and you slide it across the image. At each position, you multiply the filter values element-wise with the image patch underneath, then sum everything up. The result is a single number. Doing this at every position produces a new 2D map — the **feature map**.

A filter is essentially asking the question: "does this pattern appear here?" at every location.

### Mathematical definition

Given:
- Input $\mathbf{X}$ of shape $(H, W)$ (single channel for now)
- Kernel $\mathbf{K}$ of shape $(K, K)$
- Bias $b$ (scalar)

The output feature map $\mathbf{Z}$ at position $(i, j)$ is:

$$
\mathbf{Z}[i, j] = b + \sum_{m=0}^{K-1} \sum_{n=0}^{K-1} \mathbf{X}[i \cdot S + m,\ j \cdot S + n] \cdot \mathbf{K}[m, n]
$$

where $S$ is the stride. This is technically **cross-correlation**, not mathematical convolution (which would flip the kernel), but the ML community calls it convolution and that's what PyTorch implements.

### Worked example — from scratch

Input $\mathbf{X}$ (shape $5 \times 5$):

$$
\mathbf{X} = \begin{pmatrix}
1 & 2 & 3 & 0 & 1 \\
4 & 5 & 6 & 1 & 0 \\
7 & 8 & 9 & 2 & 1 \\
0 & 1 & 2 & 3 & 4 \\
1 & 0 & 1 & 2 & 3
\end{pmatrix}
$$

Kernel $\mathbf{K}$ (shape $3 \times 3$) — an edge detector:

$$
\mathbf{K} = \begin{pmatrix}
-1 & -1 & -1 \\
 0 &  0 &  0 \\
 1 &  1 &  1
\end{pmatrix}
$$

Bias $b = 0$, Stride $S = 1$, Padding $P = 0$.

Output size: $\lfloor (5 - 3)/1 \rfloor + 1 = 3$, so $\mathbf{Z}$ is $3 \times 3$.

**Computing $\mathbf{Z}[0,0]$** — top-left patch:

$$
\text{patch} = \begin{pmatrix}1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9\end{pmatrix}
$$

$$
\mathbf{Z}[0,0] = (1\cdot{-1} + 2\cdot{-1} + 3\cdot{-1}) + (4\cdot0 + 5\cdot0 + 6\cdot0) + (7\cdot1 + 8\cdot1 + 9\cdot1)
$$

$$
= -6 + 0 + 24 = \mathbf{18}
$$

**Computing $\mathbf{Z}[0,1]$** — shift right by 1:

$$
\text{patch} = \begin{pmatrix}2 & 3 & 0 \\ 5 & 6 & 1 \\ 8 & 9 & 2\end{pmatrix}
$$

$$
\mathbf{Z}[0,1] = (-2-3+0) + 0 + (8+9+2) = -5 + 19 = \mathbf{14}
$$

Continuing this for all 9 positions gives the full $3 \times 3$ output map.

### Output size formula

$$
\boxed{H_{out} = \left\lfloor \frac{H + 2P - K}{S} \right\rfloor + 1}
$$

$$
\boxed{W_{out} = \left\lfloor \frac{W + 2P - K}{S} \right\rfloor + 1}
$$

| Input | Kernel | Padding | Stride | Output |
|-------|--------|---------|--------|--------|
| $28 \times 28$ | $3 \times 3$ | $0$ | $1$ | $26 \times 26$ |
| $28 \times 28$ | $3 \times 3$ | $1$ | $1$ | $28 \times 28$ |
| $28 \times 28$ | $3 \times 3$ | $0$ | $2$ | $13 \times 13$ |
| $224 \times 224$ | $7 \times 7$ | $3$ | $2$ | $112 \times 112$ |

### What does a filter learn?

Different filters detect different patterns. A few classic examples:

**Horizontal edge detector:**
$$
\mathbf{K}_{hedge} = \begin{pmatrix}-1 & -1 & -1 \\ 0 & 0 & 0 \\ 1 & 1 & 1\end{pmatrix}
$$

**Vertical edge detector:**
$$
\mathbf{K}_{vedge} = \begin{pmatrix}-1 & 0 & 1 \\ -1 & 0 & 1 \\ -1 & 0 & 1\end{pmatrix}
$$

**Blur (box filter):**
$$
\mathbf{K}_{blur} = \frac{1}{9}\begin{pmatrix}1 & 1 & 1 \\ 1 & 1 & 1 \\ 1 & 1 & 1\end{pmatrix}
$$

**Sharpening:**
$$
\mathbf{K}_{sharp} = \begin{pmatrix}0 & -1 & 0 \\ -1 & 5 & -1 \\ 0 & -1 & 0\end{pmatrix}
$$

In a trained CNN, the filters are not hand-designed — they are **learned** from data via backpropagation.

---

## 5. Padding

### Why padding?

Without padding, every convolution shrinks the spatial dimensions. After many layers, the feature maps become tiny. Also, the pixels at the edges of the image contribute to fewer output values than pixels in the centre — they get "under-represented."

### Zero padding (most common)

Add $P$ rows/columns of zeros around the input before convolving:

**Input $4 \times 4$ with $P = 1$ → padded $6 \times 6$:**

$$
\mathbf{X}_{padded} = \begin{pmatrix}
0 & 0 & 0 & 0 & 0 & 0 \\
0 & x_{00} & x_{01} & x_{02} & x_{03} & 0 \\
0 & x_{10} & x_{11} & x_{12} & x_{13} & 0 \\
0 & x_{20} & x_{21} & x_{22} & x_{23} & 0 \\
0 & x_{30} & x_{31} & x_{32} & x_{33} & 0 \\
0 & 0 & 0 & 0 & 0 & 0
\end{pmatrix}
$$

### "Same" padding

The most common setting. Choose $P$ so that $H_{out} = H_{in}$ (when $S = 1$):

$$
P = \frac{K - 1}{2}
$$

For a $3 \times 3$ kernel: $P = 1$. For a $5 \times 5$ kernel: $P = 2$.

### "Valid" padding

$P = 0$ — no padding. Output shrinks every layer.

### In PyTorch

```python
# Same padding for 3x3 kernel
nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

# Valid (no padding)
nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0)

# PyTorch also supports padding_mode='reflect', 'replicate', 'circular'
```

---

## 6. Stride

### What is stride?

Stride $S$ is how many pixels the filter moves at each step. Default is $S = 1$ (move one pixel at a time).

### Stride = 1 vs Stride = 2

With $S = 1$: filters overlap heavily, output is roughly the same size as input.

With $S = 2$: filters skip every other position, output is roughly **half the size** in each dimension.

Strided convolution is often used **instead of** max pooling to downsample — it lets the network learn how to downsample rather than using a fixed operation.

### Worked example

Input $6 \times 6$, Kernel $3 \times 3$, $P = 0$:

- $S = 1$: $\lfloor(6-3)/1\rfloor + 1 = 4$ → output $4 \times 4$
- $S = 2$: $\lfloor(6-3)/2\rfloor + 1 = 2$ → output $2 \times 2$

### In PyTorch

```python
nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)  # same size
nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)  # halves size
```

---

## 7. Multiple Filters — Feature Maps

A single filter produces a single $H_{out} \times W_{out}$ feature map. But one filter can only detect one type of pattern. In practice we use $C_{out}$ different filters simultaneously.

Each filter independently slides over the input and produces its own feature map. The outputs are stacked along the channel dimension:

$$
\mathbf{Z} \in \mathbb{R}^{C_{out} \times H_{out} \times W_{out}}
$$

Each of the $C_{out}$ slices in $\mathbf{Z}$ is the response of one filter to the entire input.

**Example:** 32 filters of size $3 \times 3$ applied to a $28 \times 28$ grayscale image (with $P=1$, $S=1$):

$$
\mathbf{Z} \in \mathbb{R}^{32 \times 28 \times 28}
$$

Each of the 32 channels in $\mathbf{Z}$ detects a different pattern (different edges, textures, etc.).

### Parameter count for a conv layer

$$
\text{parameters} = C_{out} \times (C_{in} \times K \times K + 1)
$$

The $+1$ is for the bias (one bias per output channel).

**Example:** First conv layer of LeNet: $6$ filters, $5 \times 5$ kernel, $1$ input channel:

$$
6 \times (1 \times 5 \times 5 + 1) = 6 \times 26 = 156 \text{ parameters}
$$

Compare to a fully connected layer with the same input/output: $784 \times 6 = 4{,}704$ parameters.

---

## 8. Multi-Channel Convolution (RGB Images)

For a colour image with $C_{in} = 3$ channels, each filter is no longer $K \times K$ — it is $C_{in} \times K \times K = 3 \times K \times K$.

### How it works

For each of the $C_{out}$ output filters:

1. The filter has shape $(C_{in}, K, K)$
2. Each of its $C_{in}$ slices is convolved with the corresponding input channel
3. The $C_{in}$ results are **summed** to produce a single $H_{out} \times W_{out}$ feature map

$$
\mathbf{Z}[c_{out}, i, j] = b_{c_{out}} + \sum_{c_{in}=0}^{C_{in}-1} \sum_{m=0}^{K-1} \sum_{n=0}^{K-1} \mathbf{X}[c_{in},\ i \cdot S + m,\ j \cdot S + n] \cdot \mathbf{K}[c_{out}, c_{in}, m, n]
$$

So the full kernel tensor for one conv layer has shape:

$$
\mathbf{K} \in \mathbb{R}^{C_{out} \times C_{in} \times K \times K}
$$

**Example:** 32 filters, RGB input, $3 \times 3$ kernel:

$$
\mathbf{K} \in \mathbb{R}^{32 \times 3 \times 3 \times 3}
$$

$$
\text{parameters} = 32 \times (3 \times 3 \times 3 + 1) = 32 \times 28 = 896
$$

---

## 9. Activation Functions

After convolution, we apply a non-linear activation element-wise to every value in the feature map.

### ReLU (Rectified Linear Unit)

The most common activation function in CNNs:

$$
\text{ReLU}(z) = \max(0, z)
$$

$$
\text{ReLU}'(z) = \begin{cases} 1 & z > 0 \\ 0 & z \leq 0 \end{cases}
$$

- **Simple and fast** — just a threshold at zero
- **No vanishing gradient** in the positive region (derivative = 1)
- **Dead neurons** — if a neuron always outputs negative values (e.g. bad weight init), its gradient is always 0 and it never updates. This is the "dying ReLU" problem.

### Leaky ReLU

Fixes the dying ReLU problem by allowing a small gradient when $z < 0$:

$$
\text{LeakyReLU}(z) = \begin{cases} z & z > 0 \\ \alpha z & z \leq 0 \end{cases}
$$

Typically $\alpha = 0.01$.

### ELU (Exponential Linear Unit)

$$
\text{ELU}(z) = \begin{cases} z & z > 0 \\ \alpha(e^z - 1) & z \leq 0 \end{cases}
$$

Smooth at zero, negative outputs have mean closer to zero — helps with training dynamics.

### GELU (Gaussian Error Linear Unit)

Used in modern architectures (BERT, ViT):

$$
\text{GELU}(z) = z \cdot \Phi(z)
$$

where $\Phi(z)$ is the CDF of the standard normal distribution. Approximated as:

$$
\text{GELU}(z) \approx 0.5z\left(1 + \tanh\left[\sqrt{2/\pi}(z + 0.044715z^3)\right]\right)
$$

### Sigmoid and Tanh (rare in hidden layers now)

$$
\sigma(z) = \frac{1}{1+e^{-z}} \in (0, 1), \qquad \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} \in (-1, 1)
$$

Both suffer from vanishing gradients in deep networks. Sigmoid still used in output layers for binary classification.

### In PyTorch

```python
nn.ReLU()
nn.LeakyReLU(negative_slope=0.01)
nn.ELU(alpha=1.0)
nn.GELU()
torch.sigmoid(x)   # or nn.Sigmoid()
torch.tanh(x)      # or nn.Tanh()
```

---

## 10. Pooling Layers

Pooling reduces spatial dimensions, making the representation smaller, faster to compute, and more robust to small translations.

### Max Pooling

Take the **maximum** value in each pool window:

$$
\mathbf{A}_{pool}[i, j] = \max_{0 \le m < K,\ 0 \le n < K} \mathbf{A}[i \cdot S + m,\ j \cdot S + n]
$$

Most common: $2 \times 2$ max pool with $S = 2$ — **halves both spatial dimensions**.

**Example:**

$$
\mathbf{A} = \begin{pmatrix}1 & 3 & 2 & 4 \\ 5 & 6 & 1 & 2 \\ 7 & 8 & 3 & 0 \\ 4 & 2 & 1 & 9\end{pmatrix}
\xrightarrow{\text{MaxPool }2\times2, S=2}
\begin{pmatrix}6 & 4 \\ 8 & 9\end{pmatrix}
$$

- Top-left $2\times2$: $\max(1,3,5,6) = 6$
- Top-right $2\times2$: $\max(2,4,1,2) = 4$
- Bottom-left $2\times2$: $\max(7,8,4,2) = 8$
- Bottom-right $2\times2$: $\max(3,0,1,9) = 9$

### Average Pooling

Take the **mean** of each window:

$$
\mathbf{A}_{pool}[i,j] = \frac{1}{K^2} \sum_{m=0}^{K-1} \sum_{n=0}^{K-1} \mathbf{A}[i \cdot S + m,\ j \cdot S + n]
$$

### Global Average Pooling (GAP)

Collapse the entire spatial map to a single number per channel:

$$
\mathbf{A}_{GAP}[c] = \frac{1}{H \cdot W} \sum_{i=0}^{H-1} \sum_{j=0}^{W-1} \mathbf{A}[c, i, j]
$$

Used in modern architectures (ResNet, etc.) instead of flattening before the classifier. Much fewer parameters.

### Backprop through max pooling

During the forward pass, remember **which position was the max** (the "argmax"). During backprop, route the incoming gradient only to that position. All other positions get gradient $0$.

$$
\frac{\partial L}{\partial \mathbf{A}[i, j]} = \begin{cases} \delta_{pool}[i', j'] & \text{if } (i,j) \text{ was the argmax for pool position } (i', j') \\ 0 & \text{otherwise} \end{cases}
$$

### In PyTorch

```python
nn.MaxPool2d(kernel_size=2, stride=2)
nn.AvgPool2d(kernel_size=2, stride=2)
nn.AdaptiveAvgPool2d(output_size=(1, 1))  # Global Average Pooling
```

---

## 11. Batch Normalisation

### The problem: internal covariate shift

As training progresses, the distribution of each layer's inputs changes because the weights in previous layers change. This forces later layers to constantly adapt, slowing training.

### What batch norm does

For each channel $c$, normalise the activations across the batch and spatial dimensions, then rescale:

**Forward pass:**

$$
\mu_c = \frac{1}{N \cdot H \cdot W} \sum_{n,i,j} \mathbf{Z}[n, c, i, j] \qquad \text{(batch mean)}
$$

$$
\sigma_c^2 = \frac{1}{N \cdot H \cdot W} \sum_{n,i,j} \left(\mathbf{Z}[n, c, i, j] - \mu_c\right)^2 \qquad \text{(batch variance)}
$$

$$
\hat{\mathbf{Z}}[n, c, i, j] = \frac{\mathbf{Z}[n, c, i, j] - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}} \qquad \text{(normalise)}
$$

$$
\mathbf{A}_{BN}[n, c, i, j] = \gamma_c \cdot \hat{\mathbf{Z}}[n, c, i, j] + \beta_c \qquad \text{(rescale)}
$$

where:
- $\epsilon \approx 10^{-5}$ prevents division by zero
- $\gamma_c$ is a **learnable scale** parameter (one per channel)
- $\beta_c$ is a **learnable shift** parameter (one per channel)

### Why the learnable $\gamma$ and $\beta$?

Pure normalisation (without $\gamma, \beta$) would force every layer to have zero mean and unit variance, which might be the wrong representation. $\gamma$ and $\beta$ let the network undo the normalisation if that's optimal.

### At inference time

You don't have a batch — you run on a single image. So BatchNorm uses **running statistics** (exponential moving averages of $\mu$ and $\sigma^2$ accumulated during training) instead of batch statistics.

### Benefits

- Allows much higher learning rates
- Reduces sensitivity to weight initialisation
- Acts as a form of regularisation (reduces need for dropout)
- Makes the loss landscape smoother

### Where to place batch norm

The standard order is:

$$
\text{Conv} \rightarrow \text{BatchNorm} \rightarrow \text{ReLU}
$$

Some modern architectures use pre-activation: $\text{BatchNorm} \rightarrow \text{ReLU} \rightarrow \text{Conv}$

### In PyTorch

```python
nn.BatchNorm2d(num_features)  # num_features = C_out of previous conv
```

---

## 12. Dropout

### What it does

During training, randomly **zero out** neurons with probability $p$ (typically $p = 0.5$ for FC layers, $p = 0.1$-$0.2$ for conv layers):

$$
\mathbf{A}_{drop}[i] = \begin{cases} \mathbf{A}[i] / (1-p) & \text{with probability } 1-p \\ 0 & \text{with probability } p \end{cases}
$$

The division by $(1-p)$ is **inverted dropout** — it keeps the expected value of the layer's output the same regardless of $p$.

At **test time**, dropout is turned off (all neurons active, no scaling needed because of inverted dropout).

### Why it works

- Forces the network to not rely on any single neuron — more robust representations
- Effectively trains an ensemble of $2^n$ different subnetworks (where $n$ = number of neurons) and averages their predictions
- Reduces co-adaptation between neurons

### In PyTorch

```python
nn.Dropout(p=0.5)      # for fully connected layers
nn.Dropout2d(p=0.1)    # for conv layers — zeros entire feature maps
```

---

## 13. Fully Connected Layers

After the conv/pooling stack extracts spatial features, we need to make a final prediction. Fully connected (FC) layers do this.

### Flattening

Before the FC layers, the $C \times H \times W$ feature map is **flattened** to a 1D vector:

$$
\mathbf{h} = \text{flatten}(\mathbf{A}) \in \mathbb{R}^{C \cdot H \cdot W}
$$

### FC layer computation

Exactly like a standard neural network layer:

$$
\mathbf{z}_{fc} = W_{fc} \cdot \mathbf{h} + \mathbf{b}_{fc}
$$

$$
\mathbf{a}_{fc} = \text{ReLU}(\mathbf{z}_{fc})
$$

### Final classification layer

For $K$ classes, the last FC layer has $K$ output units:

$$
\mathbf{z}_{out} = W_{out} \cdot \mathbf{a}_{fc} + \mathbf{b}_{out} \in \mathbb{R}^K
$$

These raw scores are called **logits**. They are passed to softmax for probabilities.

---

## 14. The Softmax + Cross-Entropy Loss

### Softmax

Converts logits $\mathbf{z}_{out} \in \mathbb{R}^K$ to probabilities:

$$
\hat{p}_k = \text{softmax}(\mathbf{z}_{out})_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}
$$

Properties:
- $\hat{p}_k \in (0, 1)$ for all $k$
- $\sum_k \hat{p}_k = 1$
- Larger logits → larger probabilities

### Numerical stability

Computing $e^{z_k}$ can overflow for large $z_k$. The numerically stable version subtracts the max:

$$
\hat{p}_k = \frac{e^{z_k - \max_j z_j}}{\sum_{j=1}^{K} e^{z_j - \max_j z_j}}
$$

PyTorch handles this automatically.

### Cross-Entropy Loss

For a single sample with true class label $y$ (an integer index):

$$
L = -\log \hat{p}_y = -\log \frac{e^{z_y}}{\sum_j e^{z_j}}
$$

This is called **negative log-likelihood**. It's large when the predicted probability for the correct class is small, and near zero when the network is confident and correct.

For a batch of $N$ samples:

$$
L = -\frac{1}{N} \sum_{i=1}^{N} \log \hat{p}_{y_i}
$$

### Gradient of cross-entropy + softmax combined

The combined gradient is beautifully simple:

$$
\frac{\partial L}{\partial z_k} = \hat{p}_k - \mathbf{1}[k = y]
$$

where $\mathbf{1}[k = y] = 1$ if $k$ is the correct class, $0$ otherwise. In words: subtract $1$ from the probability of the true class.

This is why in PyTorch you pass raw logits (not softmax output) to `nn.CrossEntropyLoss` — it computes softmax + log + negative internally using the numerically stable log-sum-exp trick.

### Binary cross-entropy (2 classes)

$$
L = -[y \log \hat{p} + (1-y) \log(1-\hat{p})]
$$

Use `nn.BCEWithLogitsLoss` in PyTorch (more stable than computing sigmoid then `nn.BCELoss`).

---

## 15. The Full Forward Pass

Putting it all together for a complete CNN forward pass on an RGB image batch:

**Input:** $\mathbf{X} \in \mathbb{R}^{N \times 3 \times H \times W}$

**Conv Block 1:**
$$
\mathbf{Z}^{(1)} = \text{Conv}(\mathbf{X};\, \mathbf{K}^{(1)}, b^{(1)}) \in \mathbb{R}^{N \times 32 \times H_1 \times W_1}
$$
$$
\mathbf{Z}^{(1)}_{BN} = \text{BatchNorm}(\mathbf{Z}^{(1)})
$$
$$
\mathbf{A}^{(1)} = \text{ReLU}(\mathbf{Z}^{(1)}_{BN})
$$
$$
\mathbf{P}^{(1)} = \text{MaxPool}(\mathbf{A}^{(1)}) \in \mathbb{R}^{N \times 32 \times H_1/2 \times W_1/2}
$$

**Conv Block 2:**
$$
\mathbf{Z}^{(2)} = \text{Conv}(\mathbf{P}^{(1)};\, \mathbf{K}^{(2)}, b^{(2)}) \in \mathbb{R}^{N \times 64 \times H_2 \times W_2}
$$
$$
\mathbf{A}^{(2)} = \text{ReLU}(\text{BatchNorm}(\mathbf{Z}^{(2)}))
$$
$$
\mathbf{P}^{(2)} = \text{MaxPool}(\mathbf{A}^{(2)}) \in \mathbb{R}^{N \times 64 \times H_2/2 \times W_2/2}
$$

**Classifier:**
$$
\mathbf{h} = \text{flatten}(\mathbf{P}^{(2)}) \in \mathbb{R}^{N \times (64 \cdot H_2/2 \cdot W_2/2)}
$$
$$
\mathbf{h}_{drop} = \text{Dropout}(\mathbf{h})
$$
$$
\mathbf{a}_{fc} = \text{ReLU}(W_{fc} \cdot \mathbf{h}_{drop} + b_{fc})
$$
$$
\mathbf{z}_{out} = W_{out} \cdot \mathbf{a}_{fc} + b_{out} \in \mathbb{R}^{N \times K}
$$

**Loss:**
$$
L = \text{CrossEntropy}(\mathbf{z}_{out}, \mathbf{y})
$$

---

## 16. Backpropagation Through a CNN

### Overview

The same chain rule applies — we compute $\partial L / \partial W$ for every learnable weight. The tricky parts are:

1. Backprop through convolution
2. Backprop through max pooling
3. Backprop through batch norm

### 16.1 Backprop Through ReLU

ReLU's gradient is just a binary mask:

$$
\frac{\partial L}{\partial \mathbf{Z}} = \frac{\partial L}{\partial \mathbf{A}} \odot \mathbf{1}[\mathbf{Z} > 0]
$$

Where $\mathbf{1}[\mathbf{Z} > 0]$ is 1 where the pre-activation was positive, 0 elsewhere.

### 16.2 Backprop Through Convolution

Let $\delta^{out}$ be the incoming gradient (same shape as the output feature map).

**Gradient w.r.t. the kernel $\mathbf{K}$:**

The kernel gradient is itself a cross-correlation between the input $\mathbf{X}$ and the upstream gradient $\delta^{out}$:

$$
\frac{\partial L}{\partial \mathbf{K}[c_{out}, c_{in}, m, n]} = \sum_{i,j} \delta^{out}[c_{out}, i, j] \cdot \mathbf{X}[c_{in},\ i \cdot S + m,\ j \cdot S + n]
$$

In words: for each filter weight, the gradient is the sum of the upstream gradients weighted by the input patches that the weight "saw."

**Gradient w.r.t. the input $\mathbf{X}$ (to propagate further back):**

This is a **full convolution** (with padding) of the upstream gradient with the kernel flipped $180°$:

$$
\frac{\partial L}{\partial \mathbf{X}[c_{in}, i, j]} = \sum_{c_{out}} \sum_{m,n} \delta^{out}[c_{out},\ i',\ j'] \cdot \mathbf{K}[c_{out}, c_{in}, m, n]
$$

where $(i', j')$ are the output positions that "saw" input position $(i, j)$.

PyTorch's autograd handles all of this automatically — you never need to implement it manually.

**Gradient w.r.t. bias:**

$$
\frac{\partial L}{\partial b_{c_{out}}} = \sum_{i,j} \delta^{out}[c_{out}, i, j]
$$

Sum the upstream gradient over all spatial positions.

### 16.3 Backprop Through Max Pooling

During the forward pass, remember the **argmax** (which position was the max in each pool window).

During backprop, route the full upstream gradient to that winning position. All other positions get $0$:

$$
\frac{\partial L}{\partial \mathbf{A}[c, i, j]} = \begin{cases}
\delta^{pool}[c, i', j'] & \text{if } (i,j) \text{ was the argmax in window } (i', j') \\
0 & \text{otherwise}
\end{cases}
$$

### 16.4 Backprop Through Batch Norm

Let $\delta = \partial L / \partial \mathbf{A}_{BN}$ be the upstream gradient.

Gradients for the learnable parameters:

$$
\frac{\partial L}{\partial \gamma_c} = \sum_{n,i,j} \delta[n,c,i,j] \cdot \hat{\mathbf{Z}}[n,c,i,j]
$$

$$
\frac{\partial L}{\partial \beta_c} = \sum_{n,i,j} \delta[n,c,i,j]
$$

Gradient to pass back through:

$$
\frac{\partial L}{\partial \mathbf{Z}} = \frac{\gamma_c}{\sqrt{\sigma_c^2 + \epsilon}} \left[\delta - \frac{1}{m}\sum \delta - \hat{\mathbf{Z}} \cdot \frac{1}{m}\sum \delta \cdot \hat{\mathbf{Z}}\right]
$$

where $m = N \cdot H \cdot W$ is the number of elements per channel. Again, PyTorch handles all of this.

---

## 17. Weight Initialisation

### Why it matters

Bad initialisation → activations saturate or vanish → gradients vanish → training stalls.

### Xavier / Glorot Initialisation

Designed for sigmoid/tanh activations. Keeps variance roughly constant through the network:

$$
W \sim \mathcal{U}\left[-\sqrt{\frac{6}{n_{in} + n_{out}}},\ \sqrt{\frac{6}{n_{in} + n_{out}}}\right]
$$

or equivalently (normal version):

$$
W \sim \mathcal{N}\left(0,\ \frac{2}{n_{in} + n_{out}}\right)
$$

where $n_{in}$ = fan-in (input connections), $n_{out}$ = fan-out (output connections).

### He / Kaiming Initialisation

Designed for ReLU activations (accounts for the fact that ReLU kills half the inputs):

$$
W \sim \mathcal{N}\left(0,\ \frac{2}{n_{in}}\right)
$$

This is the default in PyTorch for `nn.Conv2d` and `nn.Linear`.

### In PyTorch

```python
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
nn.init.xavier_uniform_(layer.weight)
nn.init.zeros_(layer.bias)
```

---

## 18. Classic Architectures

### LeNet-5 (1998) — The first CNN

$$
\text{Input } 32\times32 \to \text{Conv}(6, 5\times5) \to \text{Pool}(2\times2) \to \text{Conv}(16, 5\times5) \to \text{Pool}(2\times2) \to \text{FC}(120) \to \text{FC}(84) \to \text{FC}(10)
$$

~60K parameters. Designed for handwritten digit recognition (MNIST).

### AlexNet (2012) — Deep learning renaissance

5 conv layers + 3 FC layers, ~60M parameters. First to use:
- ReLU (instead of tanh)
- Dropout
- Data augmentation
- GPU training

### VGGNet (2014) — Deep and simple

Key insight: use only $3 \times 3$ conv layers, but stack them deep. Two $3 \times 3$ convs have the same receptive field as one $5 \times 5$ conv, but with fewer parameters and more non-linearity.

VGG-16: 16 weight layers, ~138M parameters.

### ResNet (2015) — Skip connections

Key insight: very deep networks suffer from the **degradation problem** — adding more layers makes training harder (not just overfitting). The solution: **skip connections** (residual connections).

A residual block:

$$
\mathbf{A}_{out} = \text{ReLU}\left(\mathcal{F}(\mathbf{A}_{in}) + \mathbf{A}_{in}\right)
$$

where $\mathcal{F}$ is the conv-BN-ReLU-conv-BN stack, and $\mathbf{A}_{in}$ is added directly to the output.

The skip connection lets gradients flow directly back through the addition operation, bypassing the conv layers — solving vanishing gradients in very deep networks.

ResNet-50: 50 layers, ~25M parameters. ResNet-152: 152 layers.

### Receptive Field

The receptive field is the region of the **original input** that a neuron at a given layer "sees."

For a stack of $L$ conv layers, each with kernel size $K$ and stride $1$:

$$
\text{Receptive Field} = 1 + L \cdot (K - 1)
$$

For 3 layers of $3 \times 3$ conv: $1 + 3 \times 2 = 7 \times 7$ receptive field.

---

## 19. PyTorch — Complete Implementation

### 19.1 Building blocks

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


# ─────────────────────────────────────────────────────
# UNDERSTANDING TENSOR SHAPES
# ─────────────────────────────────────────────────────

# In PyTorch, tensors are (N, C, H, W)
# N = batch size
# C = channels
# H = height
# W = width

x = torch.randn(8, 3, 32, 32)   # 8 RGB images of 32x32
print(x.shape)   # torch.Size([8, 3, 32, 32])


# ─────────────────────────────────────────────────────
# CONV2D — DISSECTED
# ─────────────────────────────────────────────────────

conv = nn.Conv2d(
    in_channels=3,       # number of input channels (RGB = 3)
    out_channels=32,     # number of filters (output channels)
    kernel_size=3,       # 3x3 filter
    stride=1,            # move filter 1 pixel at a time
    padding=1,           # add 1 pixel of zeros on each side (same padding)
    bias=True            # add a learnable bias term (default True)
)

# Kernel shape:
print(conv.weight.shape)  # torch.Size([32, 3, 3, 3]) = (C_out, C_in, K, K)
print(conv.bias.shape)    # torch.Size([32])            = (C_out,)

# Total params in this layer:
total = 32 * 3 * 3 * 3 + 32   # = 896
print(f"Conv layer params: {total}")

out = conv(x)
print(out.shape)   # torch.Size([8, 32, 32, 32]) — same spatial size due to padding=1


# ─────────────────────────────────────────────────────
# OUTPUT SIZE CALCULATOR (useful utility)
# ─────────────────────────────────────────────────────

def conv_output_size(h_in, w_in, kernel_size, stride=1, padding=0):
    h_out = (h_in + 2 * padding - kernel_size) // stride + 1
    w_out = (w_in + 2 * padding - kernel_size) // stride + 1
    return h_out, w_out

print(conv_output_size(32, 32, kernel_size=3, padding=1, stride=1))  # (32, 32)
print(conv_output_size(32, 32, kernel_size=3, padding=0, stride=2))  # (15, 15)
print(conv_output_size(224, 224, kernel_size=7, padding=3, stride=2)) # (112, 112)
```

### 19.2 A complete CNN for CIFAR-10

```python
# ─────────────────────────────────────────────────────
# CIFAR-10: 60,000 colour images, 10 classes
# Each image: 3 x 32 x 32
# ─────────────────────────────────────────────────────

class CIFAR10_CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CIFAR10_CNN, self).__init__()

        # ── Block 1: 3 → 32 channels ─────────────────
        # Input:  (N, 3, 32, 32)
        # Output: (N, 32, 32, 32)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),    # normalise across batch, per channel
            nn.ReLU(inplace=True), # inplace saves memory
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # (N, 32, 16, 16)
            nn.Dropout2d(p=0.1)
        )

        # ── Block 2: 32 → 64 channels ────────────────
        # Input:  (N, 32, 16, 16)
        # Output: (N, 64, 8, 8)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # (N, 64, 8, 8)
            nn.Dropout2d(p=0.1)
        )

        # ── Block 3: 64 → 128 channels ───────────────
        # Input:  (N, 64, 8, 8)
        # Output: (N, 128, 4, 4)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # (N, 128, 4, 4)
            nn.Dropout2d(p=0.1)
        )

        # ── Classifier ────────────────────────────────
        # After 3 max pools: 32 → 16 → 8 → 4
        # Flattened: 128 * 4 * 4 = 2048
        self.classifier = nn.Sequential(
            nn.Flatten(),                          # (N, 128*4*4) = (N, 2048)
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)            # (N, 10) — raw logits
        )

        # Initialise weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)   # gamma = 1
                nn.init.zeros_(m.bias)    # beta = 0
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x shape: (N, 3, 32, 32)
        x = self.block1(x)    # (N, 32, 16, 16)
        x = self.block2(x)    # (N, 64, 8, 8)
        x = self.block3(x)    # (N, 128, 4, 4)
        x = self.classifier(x)# (N, 10)
        return x              # raw logits — CrossEntropyLoss handles softmax


# ── Inspect the model ────────────────────────────────────────────────────────
model = CIFAR10_CNN(num_classes=10)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Trace shapes with a dummy input
dummy = torch.randn(1, 3, 32, 32)
out = model(dummy)
print(f"Output shape: {out.shape}")  # torch.Size([1, 10])
```

### 19.3 ResNet-style residual block

```python
# ─────────────────────────────────────────────────────
# RESIDUAL BLOCK — the core of ResNet
# ─────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """
    A residual block:
        out = ReLU(F(x) + x)

    where F(x) = Conv → BN → ReLU → Conv → BN

    If in_channels != out_channels or stride != 1,
    we use a 1x1 conv (shortcut) to match dimensions.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                                kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        # Shortcut path — only needed if shapes differ
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # Main path: Conv → BN → ReLU → Conv → BN
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Skip connection: add input (possibly projected)
        out = out + self.shortcut(x)   # ← the residual addition

        # Final activation AFTER the addition
        out = self.relu(out)
        return out
```

### 19.4 Depthwise Separable Convolution (MobileNet style)

```python
# ─────────────────────────────────────────────────────
# DEPTHWISE SEPARABLE CONVOLUTION
# Reduces parameters dramatically
# ─────────────────────────────────────────────────────

class DepthwiseSeparableConv(nn.Module):
    """
    Normal conv:              C_in * C_out * K * K params
    Depthwise separable conv: C_in * K * K  +  C_in * C_out params

    For C_in=C_out=64, K=3: normal=36864, separable=4672 (~8x fewer params)
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()

        # Depthwise: one filter per input channel (groups=in_channels)
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3,
                      stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # Pointwise: 1x1 conv to combine channels
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
```

---

## 20. Training Loop in Full Detail

```python
# ─────────────────────────────────────────────────────
# DATA LOADING — CIFAR-10
# ─────────────────────────────────────────────────────

def get_data_loaders(batch_size=128):
    # Training transforms — with augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),         # crop to 32x32 after padding
        transforms.RandomHorizontalFlip(p=0.5),       # flip with 50% probability
        transforms.ColorJitter(brightness=0.2,
                               contrast=0.2,
                               saturation=0.2),        # random colour jitter
        transforms.ToTensor(),                         # (H,W,C) uint8 → (C,H,W) float [0,1]
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],             # CIFAR-10 channel means
            std =[0.2023, 0.1994, 0.2010]              # CIFAR-10 channel stds
        )
    ])

    # Test transforms — no augmentation, just normalise
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std =[0.2023, 0.1994, 0.2010]
        )
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform)

    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, test_loader


# ─────────────────────────────────────────────────────
# THE TRAINING LOOP — every line explained
# ─────────────────────────────────────────────────────

def train(model, train_loader, test_loader, num_epochs=30, lr=0.001):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    model = model.to(device)

    # Loss function
    # CrossEntropyLoss = Softmax + Log + NegativeLikelihood
    # Expects: logits (N, C) and labels (N,) as integers
    criterion = nn.CrossEntropyLoss()

    # Optimiser — Adam with weight decay (L2 regularisation)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Learning rate scheduler — reduce LR when val loss plateaus
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):

        # ── TRAINING PHASE ───────────────────────────────────────────────────
        model.train()   # CRITICAL: sets dropout and batchnorm to training mode

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):

            # Move to device
            images = images.to(device)   # (N, 3, 32, 32)
            labels = labels.to(device)   # (N,)  — integer class indices

            # ① Zero gradients from previous step
            # If you forget this, gradients ACCUMULATE across batches
            optimizer.zero_grad()

            # ② Forward pass
            logits = model(images)        # (N, 10) — raw scores

            # ③ Compute loss
            loss = criterion(logits, labels)

            # ④ Backward pass — computes ∂L/∂W for every parameter
            loss.backward()

            # ⑤ Gradient clipping — prevents exploding gradients
            # Clips gradient norm to 1.0 if it exceeds that
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # ⑥ Update weights: W ← W - η · ∂L/∂W
            optimizer.step()

            # ── Track metrics ──────────────────────────────────────────────
            train_loss += loss.item() * images.size(0)  # accumulate total loss

            # Get predicted class = argmax of logits
            _, predicted = torch.max(logits, dim=1)     # (N,)
            train_correct += (predicted == labels).sum().item()
            train_total   += labels.size(0)

            if batch_idx % 100 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs} "
                      f"Batch {batch_idx}/{len(train_loader)} "
                      f"Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / train_total
        train_acc      = train_correct / train_total

        # ── VALIDATION PHASE ─────────────────────────────────────────────────
        model.eval()    # CRITICAL: turns off dropout, uses running stats in BN

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():   # disables autograd — saves memory and compute
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                loss   = criterion(logits, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(logits, 1)
                val_correct += (predicted == labels).sum().item()
                val_total   += labels.size(0)

        avg_val_loss = val_loss / val_total
        val_acc      = val_correct / val_total

        # Step scheduler
        scheduler.step()

        # Store history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"  Val   Loss: {avg_val_loss:.4f}   | Val   Acc: {val_acc*100:.2f}%")
        print(f"  LR:         {scheduler.get_last_lr()[0]:.6f}")
        print()

    return history


# ─────────────────────────────────────────────────────
# SAVING AND LOADING CHECKPOINTS
# ─────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch, loss, path='checkpoint.pth'):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"Checkpoint saved to {path}")

def load_checkpoint(model, optimizer, path='checkpoint.pth'):
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss  = checkpoint['loss']
    print(f"Loaded checkpoint from epoch {epoch}, loss={loss:.4f}")
    return epoch, loss


# ─────────────────────────────────────────────────────
# INFERENCE ON A SINGLE IMAGE
# ─────────────────────────────────────────────────────

def predict(model, image_tensor, class_names, device='cpu'):
    model.eval()
    model = model.to(device)

    # Add batch dimension: (C, H, W) → (1, C, H, W)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)              # (1, 10)
        probs  = torch.softmax(logits, dim=1)     # (1, 10) — convert to probabilities
        conf, pred = torch.max(probs, dim=1)

    print(f"Predicted: {class_names[pred.item()]} ({conf.item()*100:.1f}% confidence)")
    return pred.item(), conf.item()


# ─────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────

if __name__ == '__main__':
    BATCH_SIZE = 128
    NUM_EPOCHS = 30
    LR         = 0.001

    train_loader, test_loader = get_data_loaders(batch_size=BATCH_SIZE)

    model   = CIFAR10_CNN(num_classes=10)
    history = train(model, train_loader, test_loader,
                    num_epochs=NUM_EPOCHS, lr=LR)

    # CIFAR-10 class names
    classes = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']
```

---

## 21. Data Augmentation

Data augmentation artificially expands your training set by applying random transforms to images. It is one of the most effective regularisation techniques for CNNs.

### Core augmentations and their purpose

| Augmentation | What it does | Why it helps |
|--------------|-------------|--------------|
| `RandomHorizontalFlip` | Flip image left/right | Objects appear at either orientation |
| `RandomCrop` | Crop a random portion | Network learns position invariance |
| `RandomRotation` | Rotate by random angle | Orientation invariance |
| `ColorJitter` | Randomly change brightness/contrast/saturation | Lighting invariance |
| `RandomErasing` | Erase a random rectangle | Occlusion robustness |
| `Normalize` | Subtract mean, divide by std | Zero-centred inputs, same scale |
| `RandomGrayscale` | Convert to grayscale with prob $p$ | Colour invariance |

### Advanced: CutMix and MixUp

**MixUp:** Blend two images and their labels:

$$
\tilde{x} = \lambda x_i + (1-\lambda) x_j, \qquad \tilde{y} = \lambda y_i + (1-\lambda) y_j
$$

**CutMix:** Paste a patch from image $j$ into image $i$:

$$
\tilde{x} = \mathbf{M} \odot x_i + (1 - \mathbf{M}) \odot x_j
$$

where $\mathbf{M}$ is a binary mask.

Both create "soft" training examples that regularise the network strongly.

```python
# Full augmentation pipeline for ImageNet-scale training
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),           # random crop and resize
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1), # brightness, contrast, sat, hue
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33))
])
```

---

## 22. Common Bugs and How to Fix Them

### Bug 1: Forgot to call `model.eval()` at test time

**Symptom:** Validation metrics are wildly inconsistent, sometimes worse than training metrics even on the training set.

**Cause:** Dropout is still active — randomly zeroing neurons during inference.

**Fix:** Always call `model.eval()` before any evaluation loop and `model.train()` before the training loop.

### Bug 2: Forgot `optimizer.zero_grad()`

**Symptom:** Loss oscillates wildly or explodes from the first batch.

**Cause:** Gradients accumulate across batches instead of being reset.

**Fix:** Call `optimizer.zero_grad()` at the start of every training iteration.

### Bug 3: Passing softmax output to `CrossEntropyLoss`

**Symptom:** Loss is very small at the start and barely decreases.

**Cause:** `CrossEntropyLoss` internally applies `log_softmax`. If you pass in softmax probabilities, it applies log_softmax to numbers already in $(0,1)$, which are then all negative, and the loss is computed on the log of small numbers — numerically wrong.

**Fix:** Pass raw **logits** to `nn.CrossEntropyLoss`. Never apply softmax before it.

### Bug 4: Wrong number of channels after flatten

**Symptom:** `RuntimeError: mat1 and mat2 shapes cannot be multiplied`

**Cause:** The spatial dimensions after pooling don't match what you hardcoded in the Linear layer.

**Fix:** Use the output size formula to compute the exact flat size, or use `nn.AdaptiveAvgPool2d(1)` to always get a fixed output size regardless of input:

```python
self.pool = nn.AdaptiveAvgPool2d(1)     # any input → (N, C, 1, 1)
self.fc   = nn.Linear(C, num_classes)   # always works
# In forward:
x = self.pool(x).flatten(1)             # (N, C)
```

### Bug 5: Not using `with torch.no_grad()` at inference

**Symptom:** Out of memory errors during validation, or much slower than expected.

**Cause:** PyTorch builds the full computation graph during forward pass (for autograd). At inference time you don't need gradients.

**Fix:** Wrap all eval code in `with torch.no_grad():`.

### Bug 6: Normalisation data leak

**Symptom:** Model performs suspiciously well on the test set.

**Cause:** You computed normalisation statistics (mean/std) over the whole dataset including the test set.

**Fix:** Compute mean and std **only on the training set**, then apply those same values to the test set.

### Bug 7: Images not in $[0, 1]$ before Normalize

**Symptom:** After normalisation, pixel values are in $[-100, 200]$ or similar absurd range.

**Cause:** `transforms.ToTensor()` converts `uint8 [0, 255]` to `float32 [0, 1]`. If you apply `Normalize` before `ToTensor`, you're subtracting 0.485 from values like 128 — completely wrong.

**Fix:** Always put `ToTensor()` before `Normalize()` in the transform pipeline.

---

## 23. Quick Reference

### Shape cheat sheet

| Layer | Input Shape | Output Shape |
|-------|------------|-------------|
| `Conv2d(Cin, Cout, K, P, S)` | $(N, C_{in}, H, W)$ | $(N, C_{out}, H_{out}, W_{out})$ |
| `BatchNorm2d(C)` | $(N, C, H, W)$ | $(N, C, H, W)$ (same) |
| `ReLU` | any | same |
| `MaxPool2d(K, S)` | $(N, C, H, W)$ | $(N, C, H/S, W/S)$ |
| `AdaptiveAvgPool2d(1)` | $(N, C, H, W)$ | $(N, C, 1, 1)$ |
| `Flatten` | $(N, C, H, W)$ | $(N, C \cdot H \cdot W)$ |
| `Linear(in, out)` | $(N, in)$ | $(N, out)$ |
| `Dropout(p)` | any | same |

### Output size formula

$$
H_{out} = \left\lfloor \frac{H_{in} + 2P - K}{S} \right\rfloor + 1
$$

### Parameter count per layer

| Layer | Parameters |
|-------|-----------|
| `Conv2d(Cin, Cout, K)` | $C_{out} \times (C_{in} \times K^2 + 1)$ |
| `BatchNorm2d(C)` | $2C$ ($\gamma$ and $\beta$) |
| `Linear(in, out)` | $in \times out + out$ |

### Training checklist

| Step | What to check |
|------|--------------|
| Data | Normalised? Augmented? Correct shape $(N, C, H, W)$? |
| Model | `model.train()` before train loop? `model.eval()` before val loop? |
| Loss | Raw logits to `CrossEntropyLoss`? Not softmax output? |
| Optimizer | `zero_grad()` before each backward? |
| Backprop | `loss.backward()` called? `optimizer.step()` after? |
| Inference | Inside `with torch.no_grad():` block? |
| Save/Load | `state_dict` used (not the whole model)? |

### Loss functions

| Problem | Loss | PyTorch |
|---------|------|---------|
| Multi-class (K classes) | Cross-entropy | `nn.CrossEntropyLoss()` |
| Binary classification | Binary cross-entropy | `nn.BCEWithLogitsLoss()` |
| Regression | Mean squared error | `nn.MSELoss()` |
| Regression (robust) | Mean absolute error | `nn.L1Loss()` |
| Regression (combined) | Huber loss | `nn.SmoothL1Loss()` |

### Optimisers

| Optimiser | Formula | When to use |
|-----------|---------|------------|
| SGD | $W \leftarrow W - \eta \nabla L$ | Large-scale, needs tuning |
| SGD + Momentum | $v \leftarrow \mu v - \eta \nabla L;\; W \leftarrow W + v$ | With LR schedule |
| Adam | Adaptive per-parameter LR | Default starting point |
| AdamW | Adam + decoupled weight decay | Better generalisation |
| RMSProp | Divide grad by running variance | RNNs, RL |

```python
optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
```

### Learning rate schedulers

```python
# Cosine annealing: smoothly decays LR from max to eta_min
optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# Step decay: multiply LR by gamma every step_size epochs
optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Reduce on plateau: reduce when val loss stops improving
optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)

# Warmup + cosine (common for transformers and large CNNs)
optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01,
                               steps_per_epoch=len(train_loader),
                               epochs=num_epochs)
```
