# Deep Learning: Complete Mathematical Reference
### Based on Géron's *Hands-On ML* — Full Derivations, No Shortcuts

---

## Table of Contents
1. [Linear Algebra Foundations](#1-linear-algebra-foundations)
2. [The Artificial Neuron](#2-the-artificial-neuron)
3. [Activation Functions](#3-activation-functions)
4. [Loss Functions](#4-loss-functions)
5. [Forward Propagation](#5-forward-propagation)
6. [Backpropagation](#6-backpropagation)
7. [Gradient Descent](#7-gradient-descent)
8. [Weight Initialization](#8-weight-initialization)
9. [Batch Normalization](#9-batch-normalization)
10. [Optimizers](#10-optimizers)
11. [Regularization](#11-regularization)
12. [Matrix Calculus Cheat Sheet](#12-matrix-calculus-cheat-sheet)

---

## 1. Linear Algebra Foundations

### 1.1 Scalars, Vectors, Matrices, Tensors

| Object | Symbol | Example Shape | Meaning in DL |
|--------|--------|---------------|---------------|
| Scalar | $x \in \mathbb{R}$ | `()` | Learning rate, loss value |
| Vector | $\mathbf{x} \in \mathbb{R}^n$ | `(n,)` | Single sample features, bias vector |
| Matrix | $\mathbf{X} \in \mathbb{R}^{m \times n}$ | `(m, n)` | Batch of samples, weight matrix |
| Tensor | $\mathcal{X} \in \mathbb{R}^{d_1 \times ... \times d_k}$ | `(d1,..,dk)` | Images, sequences |

**Convention used throughout:** Bold lowercase = vectors ($\mathbf{x}$), bold uppercase = matrices ($\mathbf{W}$), italics = scalars ($x$), superscript in brackets = layer index ($\mathbf{W}^{[l]}$), subscript = element index ($w_{ij}$).

---

### 1.2 Dot Product

For two vectors $\mathbf{a}, \mathbf{b} \in \mathbb{R}^n$:

$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i = \mathbf{a}^T \mathbf{b}$$

**Geometric interpretation:** $\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \|\mathbf{b}\| \cos\theta$

- If $\mathbf{a} \cdot \mathbf{b} > 0$: vectors point in similar directions
- If $\mathbf{a} \cdot \mathbf{b} = 0$: vectors are orthogonal (perpendicular)
- If $\mathbf{a} \cdot \mathbf{b} < 0$: vectors point in opposite directions

**In DL:** The pre-activation of a neuron is a dot product: $z = \mathbf{w} \cdot \mathbf{x} + b$

---

### 1.3 Matrix Multiplication

For $\mathbf{A} \in \mathbb{R}^{m \times k}$ and $\mathbf{B} \in \mathbb{R}^{k \times n}$, their product $\mathbf{C} = \mathbf{A}\mathbf{B} \in \mathbb{R}^{m \times n}$ has elements:

$$C_{ij} = \sum_{p=1}^{k} A_{ip} B_{pj}$$

**Shape rule:** $(m \times k) \cdot (k \times n) = (m \times n)$ — inner dimensions must match.

**For a dense layer (batch):**

$$\mathbf{Z} = \mathbf{X}\mathbf{W}^T + \mathbf{b}$$

Where:
- $\mathbf{X} \in \mathbb{R}^{m \times n_{in}}$ — batch of $m$ samples, each with $n_{in}$ features
- $\mathbf{W} \in \mathbb{R}^{n_{out} \times n_{in}}$ — weight matrix
- $\mathbf{b} \in \mathbb{R}^{n_{out}}$ — bias vector (broadcast over batch)
- $\mathbf{Z} \in \mathbb{R}^{m \times n_{out}}$ — pre-activations for entire batch

**Proof of shape:** $(m \times n_{in}) \cdot (n_{in} \times n_{out}) = (m \times n_{out})$ ✓

---

### 1.4 Norms

A norm measures the "size" of a vector or matrix.

**L1 norm (Manhattan):**
$$\|\mathbf{x}\|_1 = \sum_{i=1}^{n} |x_i|$$

**L2 norm (Euclidean) — most common:**
$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^{n} x_i^2} = \sqrt{\mathbf{x}^T\mathbf{x}}$$

**Frobenius norm (for matrices) — used in weight regularization:**
$$\|\mathbf{W}\|_F = \sqrt{\sum_{i,j} W_{ij}^2}$$

Note: $\|\mathbf{W}\|_F^2 = \sum_{i,j} W_{ij}^2$ is what actually appears in L2 regularization loss.

---

### 1.5 Transpose

For matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$, the transpose $\mathbf{A}^T \in \mathbb{R}^{n \times m}$ swaps rows and columns: $(\mathbf{A}^T)_{ij} = A_{ji}$

Key identities:
- $(\mathbf{AB})^T = \mathbf{B}^T\mathbf{A}^T$
- $(\mathbf{A}^T)^T = \mathbf{A}$
- $(\mathbf{A} + \mathbf{B})^T = \mathbf{A}^T + \mathbf{B}^T$

---

### 1.6 Broadcasting Rules

When operating on arrays of different shapes, NumPy **virtually expands** the smaller array:

1. If arrays differ in number of dimensions, pad the smaller shape with 1s on the **left**
2. Dimensions of size 1 are stretched to match the other array's dimension
3. If two dimensions differ and neither is 1 → **error**

**Example:**
```
Shape A: (4, 3)   →   (4, 3)
Shape B:    (3,)  →   (1, 3)  →  (4, 3)
Result:           →   (4, 3)
```
This is exactly how bias addition works: `Z = X @ W.T + b` where `b` has shape `(n_out,)` broadcasts over the batch.

---

## 2. The Artificial Neuron

### 2.1 The Perceptron (Linear Part)

A neuron takes $n$ inputs $x_1, x_2, ..., x_n$, computes a **weighted sum** plus a **bias**, then applies an **activation function**:

$$z = \sum_{i=1}^{n} w_i x_i + b = \mathbf{w}^T \mathbf{x} + b$$

$$a = g(z)$$

Where:
- $\mathbf{w} \in \mathbb{R}^n$ = weight vector (learned)
- $b \in \mathbb{R}$ = bias (learned)
- $z$ = pre-activation (also called *logit*)
- $g(\cdot)$ = activation function
- $a$ = output activation

**Intuition:** Each weight $w_i$ determines how much input feature $x_i$ matters. The bias $b$ allows the decision boundary to shift away from the origin.

---

### 2.2 Multi-Layer Perceptron Notation

For a network with $L$ layers (not counting input):

$$\mathbf{Z}^{[l]} = \mathbf{W}^{[l]} \mathbf{A}^{[l-1]} + \mathbf{b}^{[l]}$$
$$\mathbf{A}^{[l]} = g^{[l]}(\mathbf{Z}^{[l]})$$

**Shapes (per sample notation):**
- $\mathbf{A}^{[0]} = \mathbf{x}$ — input, shape $(n^{[0]}, 1)$
- $\mathbf{W}^{[l]}$ — shape $(n^{[l]}, n^{[l-1]})$
- $\mathbf{b}^{[l]}$ — shape $(n^{[l]}, 1)$
- $\mathbf{Z}^{[l]}, \mathbf{A}^{[l]}$ — shape $(n^{[l]}, 1)$

**Shapes (vectorized over batch of $m$ samples):**
- $\mathbf{A}^{[0]} = \mathbf{X}$ — shape $(n^{[0]}, m)$ *(column = one sample)*
- $\mathbf{W}^{[l]}$ — shape $(n^{[l]}, n^{[l-1]})$ — **same as per-sample**
- $\mathbf{b}^{[l]}$ — shape $(n^{[l]}, 1)$ — **same**, broadcasts over batch
- $\mathbf{Z}^{[l]}, \mathbf{A}^{[l]}$ — shape $(n^{[l]}, m)$

> **Note on convention:** Géron's book uses $(m, n_{out})$ (samples first, features last) which is the NumPy/Keras convention. This document uses both — clearly stated per context.

---

## 3. Activation Functions

### 3.1 Sigmoid (Logistic)

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Output range:** $(0, 1)$ — good for binary classification outputs

**Derivative derivation:**

Let $\sigma = \frac{1}{1+e^{-z}}$. Then:

$$\frac{d\sigma}{dz} = \frac{e^{-z}}{(1+e^{-z})^2}$$

We can rewrite $e^{-z} = \frac{1}{\sigma} - 1 = \frac{1-\sigma}{\sigma}$:

$$\frac{d\sigma}{dz} = \frac{1-\sigma}{\sigma} \cdot \frac{1}{(1+e^{-z})^2} \cdot \sigma^2 = \sigma(1-\sigma)$$

$$\boxed{\sigma'(z) = \sigma(z)(1 - \sigma(z))}$$

**Problem:** When $|z|$ is large, $\sigma'(z) \approx 0$. Gradients **vanish** in deep networks.

---

### 3.2 Hyperbolic Tangent (tanh)

$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$

**Output range:** $(-1, 1)$ — zero-centered (better than sigmoid for hidden layers)

**Relationship to sigmoid:**
$$\tanh(z) = 2\sigma(2z) - 1$$

**Derivative:**
$$\frac{d}{dz}\tanh(z) = 1 - \tanh^2(z) = \text{sech}^2(z)$$

**Derivation:**
$$\frac{d}{dz}\tanh(z) = \frac{(e^z+e^{-z})^2 - (e^z-e^{-z})^2}{(e^z+e^{-z})^2} = 1 - \tanh^2(z)$$

$$\boxed{\tanh'(z) = 1 - \tanh^2(z)}$$

**Still suffers from vanishing gradients** at saturating ends, but less so than sigmoid.

---

### 3.3 ReLU (Rectified Linear Unit)

$$\text{ReLU}(z) = \max(0, z) = \begin{cases} z & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}$$

**Output range:** $[0, \infty)$

**Derivative:**

$$\text{ReLU}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z < 0 \\ \text{undefined} & \text{if } z = 0 \end{cases}$$

In practice we define ReLU'(0) = 0 (subgradient).

$$\boxed{\text{ReLU}'(z) = \mathbb{1}[z > 0]}$$

**Advantages:**
- No vanishing gradient for $z > 0$
- Computationally cheap (just a threshold)
- Induces sparsity (many neurons output 0)

**Problem: Dying ReLU** — if a neuron's pre-activation is always negative, it never activates and its gradient is always 0. It "dies" and stops learning.

---

### 3.4 Leaky ReLU

$$\text{LReLU}(z) = \begin{cases} z & \text{if } z > 0 \\ \alpha z & \text{if } z \leq 0 \end{cases}$$

where $\alpha$ is a small constant, typically $\alpha = 0.01$.

$$\text{LReLU}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ \alpha & \text{if } z \leq 0 \end{cases}$$

**Fix for dying ReLU:** Negative inputs still get a small gradient $\alpha$.

**Parametric ReLU (PReLU):** $\alpha$ is learned, not fixed.

---

### 3.5 ELU (Exponential Linear Unit)

$$\text{ELU}(z) = \begin{cases} z & \text{if } z \geq 0 \\ \alpha(e^z - 1) & \text{if } z < 0 \end{cases}$$

**Derivative:**

$$\text{ELU}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ \text{ELU}(z) + \alpha & \text{if } z \leq 0 \end{cases}$$

**Why ELU > Leaky ReLU:** For very negative $z$, ELU saturates at $-\alpha$ instead of going to $-\infty$. This means the mean activation is closer to zero, reducing bias shift.

---

### 3.6 SELU (Scaled ELU)

$$\text{SELU}(z) = \lambda \begin{cases} z & \text{if } z > 0 \\ \alpha(e^z - 1) & \text{if } z \leq 0 \end{cases}$$

With specific constants:
$$\alpha \approx 1.6733, \quad \lambda \approx 1.0507$$

**Key property:** SELU is **self-normalizing** — if inputs are normalized, outputs from each layer will also be approximately normalized (mean ≈ 0, std ≈ 1). This provably prevents vanishing/exploding gradients.

**Requires:** LeCun normal initialization and no Batch Norm needed.

---

### 3.7 Softmax (Multi-class Output)

For a vector $\mathbf{z} \in \mathbb{R}^K$:

$$\text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

**Properties:**
- Output values in $(0, 1)$ and sum to 1 → valid probability distribution
- Differentiable everywhere
- Preserves relative ordering (monotone)

**Numerical stability trick (subtract max):**

$$\text{softmax}(\mathbf{z})_i = \frac{e^{z_i - \max_j z_j}}{\sum_{j=1}^{K} e^{z_j - \max_j z_j}}$$

This prevents overflow while producing identical results (numerator and denominator scaled by same constant $e^{-\max z_j}$).

**Jacobian of softmax** (needed for backprop):

$$\frac{\partial \text{softmax}(\mathbf{z})_i}{\partial z_j} = \text{softmax}(\mathbf{z})_i \cdot (\delta_{ij} - \text{softmax}(\mathbf{z})_j)$$

where $\delta_{ij}$ is the Kronecker delta (1 if $i=j$, 0 otherwise).

**Derivation:**

Case $i = j$: Using quotient rule on $\frac{e^{z_i}}{S}$ where $S = \sum_k e^{z_k}$:
$$\frac{\partial}{\partial z_i}\left(\frac{e^{z_i}}{S}\right) = \frac{e^{z_i} \cdot S - e^{z_i} \cdot e^{z_i}}{S^2} = \frac{e^{z_i}}{S}\left(1 - \frac{e^{z_i}}{S}\right) = \hat{y}_i(1 - \hat{y}_i)$$

Case $i \neq j$:
$$\frac{\partial}{\partial z_j}\left(\frac{e^{z_i}}{S}\right) = \frac{0 - e^{z_i} \cdot e^{z_j}}{S^2} = -\hat{y}_i \hat{y}_j$$

Combined: $\frac{\partial \hat{y}_i}{\partial z_j} = \hat{y}_i(\delta_{ij} - \hat{y}_j)$

---

## 4. Loss Functions

### 4.1 Mean Squared Error (MSE)

For regression with $m$ samples:

$$\mathcal{L}(\mathbf{y}, \hat{\mathbf{y}}) = \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2$$

**Gradient w.r.t. predictions:**
$$\frac{\partial \mathcal{L}}{\partial \hat{y}_i} = \frac{2}{m}(\hat{y}_i - y_i)$$

Vectorized: $\frac{\partial \mathcal{L}}{\partial \hat{\mathbf{y}}} = \frac{2}{m}(\hat{\mathbf{y}} - \mathbf{y})$

Note: Often the $\frac{1}{2}$ MSE ($= \frac{1}{2m}\sum(\hat{y}-y)^2$) is used to cancel the 2: gradient becomes $\frac{1}{m}(\hat{y}-y)$.

---

### 4.2 Mean Absolute Error (MAE)

$$\mathcal{L} = \frac{1}{m} \sum_{i=1}^{m} |y_i - \hat{y}_i|$$

**Gradient:**
$$\frac{\partial \mathcal{L}}{\partial \hat{y}_i} = \frac{1}{m} \text{sign}(\hat{y}_i - y_i)$$

**MAE vs MSE:** MAE is more robust to outliers (linear vs quadratic penalty), but gradient is constant — no acceleration near minimum.

---

### 4.3 Huber Loss

A smooth combination of MSE (near 0) and MAE (for large errors):

$$\mathcal{L}_\delta(y, \hat{y}) = \begin{cases} \frac{1}{2}(y-\hat{y})^2 & \text{if } |y-\hat{y}| \leq \delta \\ \delta \cdot |y-\hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise} \end{cases}$$

---

### 4.4 Binary Cross-Entropy (Log Loss)

For binary classification with $m$ samples, true labels $y_i \in \{0, 1\}$, predicted probabilities $\hat{y}_i \in (0, 1)$:

$$\mathcal{L} = -\frac{1}{m} \sum_{i=1}^{m} \left[ y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i) \right]$$

**Information-theoretic derivation:** This is the negative log-likelihood under a Bernoulli distribution. If model outputs $P(y=1|x) = \hat{y}$, then:

$$P(\mathbf{y}|\mathbf{X}) = \prod_{i=1}^{m} \hat{y}_i^{y_i} (1-\hat{y}_i)^{1-y_i}$$

Taking negative log: $-\log P = \mathcal{L}$ above. Minimizing BCE = maximizing likelihood.

**Gradient w.r.t. $\hat{y}_i$:**
$$\frac{\partial \mathcal{L}}{\partial \hat{y}_i} = \frac{1}{m}\left(-\frac{y_i}{\hat{y}_i} + \frac{1-y_i}{1-\hat{y}_i}\right) = \frac{1}{m}\cdot\frac{\hat{y}_i - y_i}{\hat{y}_i(1-\hat{y}_i)}$$

**Combined with sigmoid:** If $\hat{y} = \sigma(z)$, the combined gradient simplifies beautifully:

$$\frac{\partial \mathcal{L}}{\partial z_i} = \frac{\partial \mathcal{L}}{\partial \hat{y}_i} \cdot \frac{\partial \hat{y}_i}{\partial z_i} = \frac{\hat{y}_i - y_i}{\hat{y}_i(1-\hat{y}_i)} \cdot \hat{y}_i(1-\hat{y}_i) = \frac{1}{m}(\hat{y}_i - y_i)$$

$$\boxed{\frac{\partial \mathcal{L}_{BCE}}{\partial z_i} = \frac{1}{m}(\hat{y}_i - y_i)} \quad \text{(sigmoid + BCE)}$$

The saturation terms cancel! This is why sigmoid + BCE is the standard choice.

---

### 4.5 Categorical Cross-Entropy (CCE)

For multi-class classification with $K$ classes, one-hot true labels $\mathbf{y}_i \in \{0,1\}^K$:

$$\mathcal{L} = -\frac{1}{m} \sum_{i=1}^{m} \sum_{c=1}^{K} y_{ic} \log(\hat{y}_{ic})$$

Since $\mathbf{y}_i$ is one-hot (exactly one $y_{ic} = 1$), this simplifies:

$$\mathcal{L} = -\frac{1}{m} \sum_{i=1}^{m} \log(\hat{y}_{i, c_i})$$

where $c_i$ is the true class of sample $i$. We only penalize the probability assigned to the correct class.

**Combined with Softmax — the beautiful simplification:**

Let $\hat{y}_k = \text{softmax}(\mathbf{z})_k$. Then:

$$\frac{\partial \mathcal{L}}{\partial z_k} = \sum_{j=1}^{K} \frac{\partial \mathcal{L}}{\partial \hat{y}_j} \cdot \frac{\partial \hat{y}_j}{\partial z_k}$$

$$= \sum_{j=1}^{K} \left(-\frac{y_j}{\hat{y}_j}\right) \cdot \hat{y}_j(\delta_{jk} - \hat{y}_k)$$

$$= \sum_{j=1}^{K} (-y_j)(\delta_{jk} - \hat{y}_k)$$

$$= -y_k + \hat{y}_k \sum_{j=1}^{K} y_j$$

Since $\sum_{j=1}^{K} y_j = 1$ (one-hot):

$$\boxed{\frac{\partial \mathcal{L}_{CCE}}{\partial z_k} = \hat{y}_k - y_k} \quad \text{(softmax + CCE)}$$

**This is one of the most elegant results in deep learning.** The gradient of softmax+CCE is simply (prediction - truth).

---

## 5. Forward Propagation

### 5.1 Formal Algorithm

For a network with $L$ layers, batch size $m$, using **row convention** (samples as rows):

**Input:** $\mathbf{A}^{[0]} = \mathbf{X} \in \mathbb{R}^{m \times n^{[0]}}$

**For each layer $l = 1, 2, ..., L$:**

$$\mathbf{Z}^{[l]} = \mathbf{A}^{[l-1]} \mathbf{W}^{[l]T} + \mathbf{b}^{[l]} \in \mathbb{R}^{m \times n^{[l]}}$$

$$\mathbf{A}^{[l]} = g^{[l]}(\mathbf{Z}^{[l]}) \in \mathbb{R}^{m \times n^{[l]}}$$

**Output:** $\hat{\mathbf{Y}} = \mathbf{A}^{[L]}$

**Cache for backprop:** Store $\mathbf{Z}^{[l]}$ and $\mathbf{A}^{[l-1]}$ for each layer during forward pass.

### 5.2 Worked Example: 2-Layer Network

Architecture: Input(3) → Dense(4, ReLU) → Dense(2, Softmax)
Batch size: $m = 5$

```
X:   (5, 3)
W1:  (4, 3)  → Z1 = X @ W1.T + b1  → (5, 4)
b1:  (4,)
A1 = ReLU(Z1): (5, 4)
W2:  (2, 4)  → Z2 = A1 @ W2.T + b2 → (5, 2)
b2:  (2,)
A2 = Softmax(Z2): (5, 2)  ← predictions
```

---

## 6. Backpropagation

### 6.1 The Chain Rule

For composite function $f(g(x))$:

$$\frac{df}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}$$

For longer chains $f(g(h(x)))$:

$$\frac{df}{dx} = \frac{df}{dg} \cdot \frac{dg}{dh} \cdot \frac{dh}{dx}$$

**Key insight:** We compute gradients **backwards** through the chain — from output to input — reusing intermediate results. This is why it's called *backpropagation*.

---

### 6.2 Notation for Gradients

We define:

$$\delta^{[l]} = \frac{\partial \mathcal{L}}{\partial \mathbf{Z}^{[l]}}$$

This is called the "error signal" at layer $l$. Shape: $(m, n^{[l]})$.

---

### 6.3 Output Layer Backprop (Layer $L$)

**Step 1: Gradient of loss w.r.t. pre-activation $\mathbf{Z}^{[L]}$**

For softmax + CCE (or sigmoid + BCE), we derived:
$$\delta^{[L]} = \frac{1}{m}(\mathbf{A}^{[L]} - \mathbf{Y})$$

For other losses:
$$\delta^{[L]} = \frac{\partial \mathcal{L}}{\partial \mathbf{A}^{[L]}} \odot g'^{[L]}(\mathbf{Z}^{[L]})$$

where $\odot$ = element-wise multiplication.

**Step 2: Gradient w.r.t. weights:**
$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[L]}} = (\delta^{[L]})^T \mathbf{A}^{[L-1]}$$

Shape check: $(\delta^{[L]})^T \in \mathbb{R}^{n^{[L]} \times m}$ times $\mathbf{A}^{[L-1]} \in \mathbb{R}^{m \times n^{[L-1]}}$ = $\mathbb{R}^{n^{[L]} \times n^{[L-1]}}$ = shape of $\mathbf{W}^{[L]}$ ✓

**Step 3: Gradient w.r.t. bias:**
$$\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[L]}} = \sum_{i=1}^{m} \delta^{[L]}_i = \text{sum over batch (axis=0)}$$

Shape: $(n^{[L]},)$ ✓

**Step 4: Pass gradient to previous layer:**
$$\frac{\partial \mathcal{L}}{\partial \mathbf{A}^{[L-1]}} = \delta^{[L]} \mathbf{W}^{[L]}$$

Shape: $(m, n^{[L]}) \cdot (n^{[L]}, n^{[L-1]}) = (m, n^{[L-1]})$ ✓

---

### 6.4 General Hidden Layer $l$ Backprop

Given $\frac{\partial \mathcal{L}}{\partial \mathbf{A}^{[l]}}$ from the layer above:

**Step 1: Error signal at layer $l$:**
$$\delta^{[l]} = \frac{\partial \mathcal{L}}{\partial \mathbf{A}^{[l]}} \odot g'^{[l]}(\mathbf{Z}^{[l]})$$

**Step 2–4 same as output layer (replacing $L$ with $l$).**

### 6.5 Complete Backprop Algorithm

```
Forward pass:
  cache Z[l], A[l-1] for each layer l

Backward pass:
  dA_out = dL/dA[L]           (from loss)

  For l = L, L-1, ..., 1:
    dZ = dA_out ⊙ g'[l](Z[l])   (or use combined formula for output layer)
    dW[l] = (1/m) * dZ.T @ A[l-1]
    db[l] = (1/m) * sum(dZ, axis=0)
    dA_out = dZ @ W[l]          (pass back to previous layer)
```

---

### 6.6 Full Derivation: 2-Layer Network, MSE + Linear Output

Architecture: Dense(ReLU) → Dense(Linear), Loss = MSE

Forward:
- $\mathbf{Z}^{[1]} = \mathbf{X}\mathbf{W}^{[1]T} + \mathbf{b}^{[1]}$
- $\mathbf{A}^{[1]} = \text{ReLU}(\mathbf{Z}^{[1]})$
- $\mathbf{Z}^{[2]} = \mathbf{A}^{[1]}\mathbf{W}^{[2]T} + \mathbf{b}^{[2]}$
- $\hat{\mathbf{Y}} = \mathbf{Z}^{[2]}$ (linear activation)
- $\mathcal{L} = \frac{1}{m}\|\hat{\mathbf{Y}} - \mathbf{Y}\|_F^2$

Backward:

$$\frac{\partial \mathcal{L}}{\partial \hat{\mathbf{Y}}} = \frac{2}{m}(\hat{\mathbf{Y}} - \mathbf{Y})$$

$$\delta^{[2]} = \frac{2}{m}(\hat{\mathbf{Y}} - \mathbf{Y})$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[2]}} = (\delta^{[2]})^T \mathbf{A}^{[1]}, \quad \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[2]}} = \sum \delta^{[2]}$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{A}^{[1]}} = \delta^{[2]} \mathbf{W}^{[2]}$$

$$\delta^{[1]} = \frac{\partial \mathcal{L}}{\partial \mathbf{A}^{[1]}} \odot \mathbb{1}[\mathbf{Z}^{[1]} > 0] \quad \text{(ReLU derivative)}$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[1]}} = (\delta^{[1]})^T \mathbf{X}, \quad \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[1]}} = \sum \delta^{[1]}$$

---

## 7. Gradient Descent

### 7.1 The Update Rule

The fundamental parameter update:

$$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta \nabla_{\boldsymbol{\theta}} \mathcal{L}$$

where $\eta > 0$ is the **learning rate**.

**Why subtract the gradient?** The gradient $\nabla \mathcal{L}$ points in the direction of steepest *increase* of the loss. We want to *decrease* the loss, so we move opposite to the gradient.

---

### 7.2 Variants

| Variant | Gradient Computed On | Update Per | Noise Level |
|---------|---------------------|------------|-------------|
| Batch GD | Entire dataset | Epoch | None (exact) |
| SGD | 1 sample | Sample | High |
| Mini-batch GD | $B$ samples | Mini-batch | Medium |

**Mini-batch is standard.** Common batch sizes: 32, 64, 128, 256.

**Key formulas — mini-batch:**

1. Sample a batch $\mathcal{B} \subset \{1,...,m\}$ of size $B$
2. Forward pass on batch: compute $\hat{\mathbf{Y}}$
3. Compute loss $\mathcal{L}_\mathcal{B} = \frac{1}{B}\sum_{i \in \mathcal{B}} \ell_i$
4. Backprop: compute $\nabla_{\mathbf{W}^{[l]}} \mathcal{L}_\mathcal{B}$, $\nabla_{\mathbf{b}^{[l]}} \mathcal{L}_\mathcal{B}$
5. Update: $\mathbf{W}^{[l]} \leftarrow \mathbf{W}^{[l]} - \eta \nabla_{\mathbf{W}^{[l]}} \mathcal{L}_\mathcal{B}$

**One epoch** = one full pass through the dataset = $\lceil m/B \rceil$ mini-batch updates.

---

### 7.3 Learning Rate Effects

- **Too large $\eta$:** Gradient descent diverges — overshoots the minimum
- **Too small $\eta$:** Extremely slow convergence
- **Optimal $\eta$:** Fast, stable convergence

**Learning Rate Schedules:**
- Step decay: $\eta_t = \eta_0 \cdot \text{drop}^{\lfloor t/\text{step\_size} \rfloor}$
- Exponential decay: $\eta_t = \eta_0 \cdot e^{-\lambda t}$
- 1-cycle policy: warm-up then cooldown
- Cosine annealing: $\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max}-\eta_{min})(1 + \cos(\pi t/T))$

---

## 8. Weight Initialization

### 8.1 Why It Matters

Bad initialization → vanishing or exploding activations/gradients → network can't learn.

**Thought experiment:** If all weights are the same value, all neurons in a layer compute identical outputs and receive identical gradients. They can never learn different features — symmetry is never broken. **Never initialize with constant values (except biases to 0).**

**Too large:** $\sigma(Wx) \approx 0$ or $1$ → gradients $\approx 0$ (vanish)

**Too small:** Activations shrink each layer → gradients also shrink (vanish)

**Goal:** Initialize so that signal **neither vanishes nor explodes** as it propagates forward and backward.

---

### 8.2 Xavier / Glorot Initialization

Designed for **sigmoid** and **tanh** (linear regime near 0).

The key insight: for a layer with $n_{in}$ inputs and $n_{out}$ outputs, we want:
$$\text{Var}(z) \approx \text{Var}(x) \quad \text{(variance preserved)}$$

This requires: $\text{Var}(w) = \frac{1}{n_{in}}$

The Glorot paper uses a compromise between forward and backward propagation:

$$\text{Var}(w) = \frac{2}{n_{in} + n_{out}}$$

**Two forms:**

Normal: $w \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in}+n_{out}}}\right)$

Uniform: $w \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{in}+n_{out}}}, \sqrt{\frac{6}{n_{in}+n_{out}}}\right)$

---

### 8.3 He Initialization

Designed for **ReLU** activations. Since ReLU zeros out half the inputs (on average), the effective fan-in is halved. To preserve variance:

$$\text{Var}(w) = \frac{2}{n_{in}}$$

Normal: $w \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in}}}\right)$

**In PyTorch/Keras:** This is called `he_normal` or `he_uniform`.

---

### 8.4 LeCun Initialization

Designed for **SELU**:

$$\text{Var}(w) = \frac{1}{n_{in}}$$

Normal: $w \sim \mathcal{N}\left(0, \sqrt{\frac{1}{n_{in}}}\right)$

| Activation | Initialization | Std |
|-----------|----------------|-----|
| tanh/sigmoid | Glorot | $\sqrt{2/(n_{in}+n_{out})}$ |
| ReLU / Leaky ReLU / ELU | He | $\sqrt{2/n_{in}}$ |
| SELU | LeCun | $\sqrt{1/n_{in}}$ |

---

## 9. Batch Normalization

### 9.1 Motivation: Internal Covariate Shift

As training proceeds, the distribution of inputs to each layer keeps shifting because the parameters of earlier layers change. This forces each layer to constantly adapt to new distributions — slowing training. This is called *internal covariate shift*.

**Solution:** Normalize the input to each layer to have mean 0 and variance 1.

---

### 9.2 Forward Pass Algorithm

For a mini-batch $\mathcal{B} = \{x_1, ..., x_m\}$:

**Step 1: Compute mini-batch mean:**
$$\mu_\mathcal{B} = \frac{1}{m} \sum_{i=1}^{m} x_i$$

**Step 2: Compute mini-batch variance:**
$$\sigma_\mathcal{B}^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_\mathcal{B})^2$$

**Step 3: Normalize:**
$$\hat{x}_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}}$$

where $\epsilon \approx 10^{-5}$ prevents division by zero.

**Step 4: Scale and shift (learnable):**
$$y_i = \gamma \hat{x}_i + \beta$$

Parameters $\gamma$ (scale) and $\beta$ (shift) are **learned by backprop**. They allow the network to "undo" the normalization if needed (e.g., learn to output non-zero mean).

---

### 9.3 Backpropagation Through Batch Norm

Given upstream gradient $\frac{\partial \mathcal{L}}{\partial y_i}$:

$$\frac{\partial \mathcal{L}}{\partial \gamma} = \sum_{i=1}^{m} \frac{\partial \mathcal{L}}{\partial y_i} \cdot \hat{x}_i$$

$$\frac{\partial \mathcal{L}}{\partial \beta} = \sum_{i=1}^{m} \frac{\partial \mathcal{L}}{\partial y_i}$$

$$\frac{\partial \mathcal{L}}{\partial \hat{x}_i} = \frac{\partial \mathcal{L}}{\partial y_i} \cdot \gamma$$

$$\frac{\partial \mathcal{L}}{\partial \sigma_\mathcal{B}^2} = \sum_{i=1}^{m} \frac{\partial \mathcal{L}}{\partial \hat{x}_i} \cdot (x_i - \mu_\mathcal{B}) \cdot \left(-\frac{1}{2}\right) (\sigma_\mathcal{B}^2 + \epsilon)^{-3/2}$$

$$\frac{\partial \mathcal{L}}{\partial \mu_\mathcal{B}} = \left(\sum_{i=1}^{m} \frac{\partial \mathcal{L}}{\partial \hat{x}_i} \cdot \frac{-1}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}}\right) + \frac{\partial \mathcal{L}}{\partial \sigma_\mathcal{B}^2} \cdot \frac{-2}{m} \sum_{i=1}^{m}(x_i - \mu_\mathcal{B})$$

$$\frac{\partial \mathcal{L}}{\partial x_i} = \frac{\partial \mathcal{L}}{\partial \hat{x}_i} \cdot \frac{1}{\sqrt{\sigma_\mathcal{B}^2+\epsilon}} + \frac{\partial \mathcal{L}}{\partial \sigma_\mathcal{B}^2} \cdot \frac{2(x_i-\mu_\mathcal{B})}{m} + \frac{\partial \mathcal{L}}{\partial \mu_\mathcal{B}} \cdot \frac{1}{m}$$

---

### 9.4 Training vs Inference

During **training:** Use batch statistics $\mu_\mathcal{B}$, $\sigma_\mathcal{B}^2$

During **inference:** Use **running averages** accumulated during training:
$$\mu_{run} \leftarrow (1-\rho)\mu_{run} + \rho \mu_\mathcal{B}$$
$$\sigma^2_{run} \leftarrow (1-\rho)\sigma^2_{run} + \rho \sigma^2_\mathcal{B}$$

where $\rho$ is the momentum (typically 0.99).

---

## 10. Optimizers

### 10.1 Vanilla SGD

$$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta \mathbf{g}_t$$

where $\mathbf{g}_t = \nabla_{\boldsymbol{\theta}} \mathcal{L}_t$.

**Problem:** Slow convergence, can oscillate in narrow valleys, sensitivity to learning rate.

---

### 10.2 SGD with Momentum

Accumulates velocity in directions of persistent gradient:

$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} - \eta \mathbf{g}_t$$
$$\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} + \mathbf{v}_t$$

where $\beta \in [0, 1)$ is the momentum coefficient (typically 0.9).

**Physical analogy:** A ball rolling down a hill accumulates momentum. In flat directions with consistent gradient, velocity builds up. In oscillating directions, velocity cancels out.

**Effective learning rate:** In steady state (constant gradient), velocity converges to $\mathbf{v} = -\frac{\eta}{1-\beta}\mathbf{g}$. With $\beta=0.9$, effective learning rate is $10\times$ the nominal rate.

---

### 10.3 Nesterov Accelerated Gradient (NAG)

"Look ahead" before computing gradient:

$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} - \eta \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_{t-1} + \beta \mathbf{v}_{t-1})$$
$$\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} + \mathbf{v}_t$$

Instead of computing gradient at current position, compute it at the **lookahead position** $\boldsymbol{\theta} + \beta\mathbf{v}$. This gives slightly better convergence as it "corrects" course before overshooting.

---

### 10.4 AdaGrad (Adaptive Gradient)

Adapts learning rate **per parameter** based on accumulated squared gradients:

$$\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t \odot \mathbf{g}_t$$
$$\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t} + \epsilon} \odot \mathbf{g}_t$$

**Intuition:** Parameters that received large gradients in the past get smaller learning rates (they may already be near minimum). Rare/infrequent parameters get larger steps.

**Problem:** $\mathbf{s}_t$ only ever grows → learning rate eventually becomes infinitesimally small → learning stops prematurely. Good for sparse features; bad for deep networks.

---

### 10.5 RMSProp

Fixes AdaGrad's shrinking learning rate by using **exponential moving average** of squared gradients:

$$\mathbf{s}_t = \rho \mathbf{s}_{t-1} + (1-\rho) \mathbf{g}_t \odot \mathbf{g}_t$$
$$\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t} + \epsilon} \odot \mathbf{g}_t$$

Typical: $\rho = 0.9$, $\epsilon = 10^{-8}$, $\eta = 0.001$

$\rho$ controls "memory" — high $\rho$ = slow-moving average.

---

### 10.6 Adam (Adaptive Moment Estimation)

Adam combines momentum (first moment) and RMSProp (second moment):

**First moment (momentum):**
$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1) \mathbf{g}_t$$

**Second moment (uncentered variance):**
$$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2) \mathbf{g}_t \odot \mathbf{g}_t$$

**Bias correction** (important for early timesteps — $\mathbf{m}_0 = \mathbf{v}_0 = 0$ biases toward 0):
$$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1-\beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1-\beta_2^t}$$

**Update:**
$$\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \frac{\eta}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \odot \hat{\mathbf{m}}_t$$

**Typical defaults:** $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$, $\eta=0.001$

**Why bias correction?** At $t=1$: $\mathbf{m}_1 = (1-\beta_1)\mathbf{g}_1 = 0.1\mathbf{g}_1$. Without correction, first moment is 10× too small. Correcting by $1/(1-\beta_1^1) = 10$ fixes this.

---

### 10.7 AdaMax

Variant of Adam using L∞ norm instead of L2:

$$\mathbf{u}_t = \max(\beta_2 \mathbf{u}_{t-1}, |\mathbf{g}_t|)$$
$$\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \frac{\eta}{(1-\beta_1^t)\mathbf{u}_t} \odot \mathbf{m}_t$$

Note: No bias correction needed for $\mathbf{u}_t$ (max operation not biased toward 0).

---

### 10.8 Nadam (Nesterov + Adam)

Adam with Nesterov lookahead for the momentum term:

$$\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \frac{\eta}{\sqrt{\hat{\mathbf{v}}_t}+\epsilon} \left(\beta_1\hat{\mathbf{m}}_t + \frac{(1-\beta_1)\mathbf{g}_t}{1-\beta_1^t}\right)$$

---

## 11. Regularization

### 11.1 Bias-Variance Tradeoff

**Total error = Bias² + Variance + Irreducible noise**

- **High bias (underfitting):** Model too simple, can't capture data structure. Fix: more layers/neurons, better features.
- **High variance (overfitting):** Model memorizes training data, fails to generalize. Fix: regularization, more data, simpler model.

---

### 11.2 L2 Regularization (Weight Decay)

Add a penalty term to the loss:

$$\mathcal{L}_{reg} = \mathcal{L} + \frac{\lambda}{2m} \sum_{l=1}^{L} \|\mathbf{W}^{[l]}\|_F^2$$

**Gradient update:**

$$\frac{\partial \mathcal{L}_{reg}}{\partial \mathbf{W}^{[l]}} = \frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}} + \frac{\lambda}{m}\mathbf{W}^{[l]}$$

Parameter update becomes:

$$\mathbf{W}^{[l]} \leftarrow \mathbf{W}^{[l]} - \eta\left(\nabla \mathcal{L} + \frac{\lambda}{m}\mathbf{W}^{[l]}\right) = \left(1 - \frac{\eta\lambda}{m}\right)\mathbf{W}^{[l]} - \eta\nabla\mathcal{L}$$

The factor $\left(1-\frac{\eta\lambda}{m}\right) < 1$ **shrinks** the weights each step — hence "weight decay".

**Effect:** Prefers small weights → simpler functions → less overfitting.

**Note:** Biases are not regularized (they don't contribute to complexity in the same way).

---

### 11.3 L1 Regularization

$$\mathcal{L}_{reg} = \mathcal{L} + \frac{\lambda}{m} \sum_{l=1}^{L} \|\mathbf{W}^{[l]}\|_1$$

**Gradient:**

$$\frac{\partial \mathcal{L}_{reg}}{\partial \mathbf{W}^{[l]}} = \frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}} + \frac{\lambda}{m}\text{sign}(\mathbf{W}^{[l]})$$

**L1 vs L2:**
- L1 promotes **sparsity** — many weights become exactly 0 (automatic feature selection)
- L2 makes all weights small but non-zero
- L1 gradient is constant regardless of weight magnitude → keeps pushing weights to 0
- L2 gradient is proportional to weight → small weights get tiny gradient, don't reach 0

---

### 11.4 Dropout

During training, each neuron is **randomly disabled** with probability $p$ (drop probability):

$$\mathbf{a}^{[l]} = \frac{1}{1-p} \cdot \mathbf{m} \odot \mathbf{a}^{[l]}$$

where $\mathbf{m} \sim \text{Bernoulli}(1-p)$ is a binary mask (1 = keep, 0 = drop), and $\frac{1}{1-p}$ is the **inverted dropout scale factor**.

**Why the scale factor?** At test time we use all neurons (no dropout). To keep expected activation magnitude the same as training, we scale up during training: $\mathbb{E}[\tilde{a}] = (1-p) \cdot \frac{1}{1-p} \cdot a = a$.

**Alternatively:** Scale down by $(1-p)$ at test time (but inverted dropout is more common).

**Gradient through dropout:**
$$\frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[l-1]}} = \frac{1}{1-p} \cdot \mathbf{m} \odot \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[l]}}$$

The same mask $\mathbf{m}$ used in the forward pass is reused in backprop.

**Why it works:** Forces the network to not rely on any single neuron. Each neuron must learn robust features independently. Effectively trains an ensemble of $2^N$ different thinned networks.

---

### 11.5 Gradient Clipping

Prevents exploding gradients (especially in RNNs):

**Clip by value:**
$$g_i \leftarrow \max(-\theta, \min(\theta, g_i))$$

**Clip by norm:**
$$\mathbf{g} \leftarrow \mathbf{g} \cdot \frac{\theta}{\max(\theta, \|\mathbf{g}\|_2)}$$

Clip by norm preserves gradient direction; clip by value can change it.

---

### 11.6 Max-Norm Regularization

After each gradient update, clip each neuron's incoming weight vector:

$$\mathbf{w} \leftarrow \mathbf{w} \cdot \min\left(1, \frac{r}{\|\mathbf{w}\|_2}\right)$$

Ensures $\|\mathbf{w}\|_2 \leq r$ at all times. Works well with dropout.

---

## 12. Matrix Calculus Cheat Sheet

### 12.1 Scalar by Vector

If $f: \mathbb{R}^n \to \mathbb{R}$, then $\nabla_{\mathbf{x}} f = \frac{\partial f}{\partial \mathbf{x}} \in \mathbb{R}^n$ where $\left(\frac{\partial f}{\partial \mathbf{x}}\right)_i = \frac{\partial f}{\partial x_i}$

| $f(\mathbf{x})$ | $\frac{\partial f}{\partial \mathbf{x}}$ |
|-----------------|------------------------------------------|
| $\mathbf{a}^T\mathbf{x}$ | $\mathbf{a}$ |
| $\mathbf{x}^T\mathbf{a}$ | $\mathbf{a}$ |
| $\mathbf{x}^T\mathbf{x}$ | $2\mathbf{x}$ |
| $\mathbf{x}^T\mathbf{A}\mathbf{x}$ | $(\mathbf{A}+\mathbf{A}^T)\mathbf{x}$ |
| $\|\mathbf{x}\|^2$ | $2\mathbf{x}$ |

### 12.2 Scalar by Matrix

If $f: \mathbb{R}^{m \times n} \to \mathbb{R}$, then $\frac{\partial f}{\partial \mathbf{W}} \in \mathbb{R}^{m \times n}$

| $f(\mathbf{W})$ | $\frac{\partial f}{\partial \mathbf{W}}$ |
|-----------------|------------------------------------------|
| $\text{tr}(\mathbf{W})$ | $\mathbf{I}$ |
| $\text{tr}(\mathbf{AW})$ | $\mathbf{A}^T$ |
| $\|\mathbf{W}\|_F^2$ | $2\mathbf{W}$ |
| $\mathbf{a}^T\mathbf{W}\mathbf{b}$ | $\mathbf{a}\mathbf{b}^T$ |

### 12.3 Key Rules

**Product rule:** $\nabla(\mathbf{f}^T\mathbf{g}) = (\nabla\mathbf{f}^T)\mathbf{g} + (\nabla\mathbf{g}^T)\mathbf{f}$

**Chain rule:** If $f = f(g(\mathbf{x}))$, then $\nabla_\mathbf{x} f = \frac{\partial g}{\partial \mathbf{x}}^T \nabla_g f$

### 12.4 NumPy ↔ Math Notation

| Math | NumPy |
|------|-------|
| $\mathbf{A}\mathbf{B}$ | `A @ B` |
| $\mathbf{A}^T$ | `A.T` |
| $\mathbf{a} \cdot \mathbf{b}$ | `np.dot(a, b)` |
| $\mathbf{a} \odot \mathbf{b}$ (element-wise) | `a * b` |
| $\|\mathbf{x}\|_2$ | `np.linalg.norm(x)` |
| $\|\mathbf{W}\|_F$ | `np.linalg.norm(W, 'fro')` |
| $\sum_i x_i$ | `np.sum(x)` |
| $\sum_i x_i$ (over batch, axis 0) | `np.sum(x, axis=0)` |
| $\frac{1}{m}\sum_i x_i$ | `np.mean(x, axis=0)` |
| $\text{sign}(\mathbf{x})$ | `np.sign(x)` |
| $e^{\mathbf{x}}$ (element-wise) | `np.exp(x)` |
| $\log(\mathbf{x})$ (element-wise) | `np.log(x)` |
| $\max(0, \mathbf{x})$ | `np.maximum(0, x)` |
| $\mathbb{1}[x > 0]$ | `(x > 0).astype(float)` |
| $\text{clip}(x, a, b)$ | `np.clip(x, a, b)` |
| $\text{diag}(\mathbf{v})$ | `np.diag(v)` |
| $\text{tr}(\mathbf{A})$ | `np.trace(A)` |

---

*These notes cover all the math needed from Chapters 10–12 of Géron's book and lay the groundwork for Chapters 13–19. For each chapter going forward, refer to the corresponding section in the NumPy notebook for implementation.*
