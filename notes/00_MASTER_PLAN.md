# AI and ML for Coders in PyTorch - Complete Study Notes

## Master Plan

**Book**: *AI and ML for Coders in PyTorch: A Coder's Guide to Generative AI and Machine Learning*
**Author**: Laurence Moroney | **Foreword**: Andrew Ng
**Publisher**: O'Reilly Media, 2025 | **Pages**: 445 (20 Chapters)

---

## How These Notes Are Organized

The book's 20 chapters are divided into **8 Phases**, each producing its own detailed markdown file. Every phase contains:

- **Conceptual explanations** - what each idea means, in plain language
- **Full mathematics** - every equation derived step-by-step, no hand-waving
- **Code walkthroughs** - annotated PyTorch code with line-by-line explanation
- **Diagrams described** - architecture diagrams, data flow, and intuition-building visuals
- **Key terminology** - bolded and defined on first use
- **Summary & review questions** - to test your understanding

---

## Phase Map

| Phase | Chapters | Title | Status |
|-------|----------|-------|--------|
| **1** | Ch 1 | **Foundations: What is ML & Getting Started with PyTorch** | **Done** |
| **2** | Ch 2-3 | **Computer Vision: From Dense Networks to CNNs** | **Done** |
| **3** | Ch 4 | **Working with Public Datasets (TorchVision & Beyond)** | **Done** |
| **4** | Ch 5-8 | **Natural Language Processing: Text to Numbers** | Pending |
| **5** | Ch 9-11 | **Sequence Models: RNNs, LSTMs & Text Generation** | Pending |
| **6** | Ch 12-13 | **Time-Series Forecasting with Neural Networks** | Pending |
| **7** | Ch 14-16 | **Transformers, Hugging Face & Fine-Tuning LLMs** | Pending |
| **8** | Ch 17-20 | **Generative AI: Ollama, RAG, Diffusers & LoRA** | Pending |

---

## Phase 1 - Foundations: What is ML & Getting Started with PyTorch
**Covers**: Chapter 1 (Pages 1-21)

### Topics:
1. What is Machine Learning? - The paradigm shift from traditional programming
2. Limitations of Traditional Programming - why rules break down
3. From Programming to Learning - the ML mental model
4. What is PyTorch? - history, ecosystem, why it won
5. Installing & Using PyTorch - Python, PyCharm, Google Colab
6. Getting Started with Machine Learning - your first neural network
7. **Math**: Linear functions, loss functions, gradient descent, optimizers
8. Seeing What the Network Learned - interpreting weights and biases

### Key Math Topics:
- `y = mx + c` (linear model)
- Loss/Cost functions (MSE)
- Gradient Descent (partial derivatives, learning rate)
- Backpropagation (chain rule)
- Tensors and tensor operations

---

## Phase 2 - Computer Vision: From Dense Networks to CNNs
**Covers**: Chapters 2-3 (Pages 23-78)

### Topics:
1. How Computer Vision works in ML
2. Fashion MNIST dataset - loading, exploring, normalizing
3. Neurons for Vision - dense/fully-connected layers
4. Designing & training a neural network for image classification
5. Overfitting and Early Stopping
6. **Convolutions** - what they are, how filters detect features
7. **Pooling** - max pooling, average pooling, dimensionality reduction
8. Building CNNs in PyTorch
9. Horses vs Humans - a real binary classification problem
10. Image Augmentation - transforms for generalization
11. Transfer Learning - leveraging pretrained models
12. Multiclass Classification
13. Dropout Regularization

### Key Math Topics:
- Activation functions (ReLU, Softmax, Sigmoid)
- Cross-entropy loss (binary and categorical)
- Convolution operation (kernel * input, stride, padding)
- Pooling mathematics
- Softmax derivation: `softmax(z_i) = e^{z_i} / sum(e^{z_j})`
- Parameter counting in networks

---

## Phase 3 - Working with Public Datasets
**Covers**: Chapter 4 (Pages 79-100)

### Topics:
1. Using TorchVision Datasets
2. Using TensorFlow Datasets with PyTorch
3. Data loading pipelines - DataLoader, Dataset, transforms
4. Batching, shuffling, and preprocessing
5. Loading custom datasets
6. ETL patterns for ML data

### Key Concepts:
- DataLoader mechanics (batching, num_workers, pin_memory)
- Transform pipelines (Compose, Normalize, ToTensor)
- Train/Validation/Test splits

---

## Phase 4 - Natural Language Processing: Text to Numbers
**Covers**: Chapters 5-8 (Pages 101-186)

### Topics:
1. Encoding language as numbers - tokenization
2. Word frequency and sequence encoding
3. Making sentiment programmable - sentiment analysis
4. Building text classifiers
5. Word embeddings - what they are and why they matter
6. Using pretrained embeddings (Word2Vec, GloVe)
7. RNNs for text classification
8. Padding and truncating sequences

### Key Math Topics:
- One-hot encoding
- Embedding vectors and embedding spaces
- Dot product similarity / Cosine similarity
- TF-IDF (Term Frequency - Inverse Document Frequency)
- RNN cell equations: `h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b)`

---

## Phase 5 - Sequence Models: RNNs, LSTMs & Text Generation
**Covers**: Chapters 9-11 (Pages 187-248)

### Topics:
1. Understanding sequence-to-sequence models
2. LSTM architecture - gates, cell state, hidden state
3. Bidirectional RNNs
4. Stacked/Deep RNNs
5. Text generation with character/word-level models
6. Combining CNNs with RNNs for sequences
7. Convolutional methods for sequence data (1D convolutions)

### Key Math Topics:
- LSTM equations (all four gates):
  - Forget gate: `f_t = sigma(W_f . [h_{t-1}, x_t] + b_f)`
  - Input gate: `i_t = sigma(W_i . [h_{t-1}, x_t] + b_i)`
  - Candidate: `C_tilde = tanh(W_C . [h_{t-1}, x_t] + b_C)`
  - Cell state: `C_t = f_t * C_{t-1} + i_t * C_tilde`
  - Output gate: `o_t = sigma(W_o . [h_{t-1}, x_t] + b_o)`
  - Hidden state: `h_t = o_t * tanh(C_t)`
- GRU equations (simplified LSTM)
- Temperature in text generation (softmax with temperature)
- 1D convolution for sequences

---

## Phase 6 - Time-Series Forecasting with Neural Networks
**Covers**: Chapters 12-13 (Pages 249-296)

### Topics:
1. What is a time series? Components: trend, seasonality, noise
2. Naive forecasting and statistical baselines
3. Fixed vs adaptive windowing
4. Moving averages and differencing
5. Neural networks for time-series
6. DNN approach
7. RNN/LSTM approach for time-series
8. Evaluation metrics: MAE, MSE, RMSE, MAPE

### Key Math Topics:
- Moving average: `MA_t = (1/k) * sum(x_{t-i})` for i=0..k-1
- Windowed dataset creation for supervised learning
- MAE: `(1/n) * sum(|y_i - y_hat_i|)`
- MSE: `(1/n) * sum((y_i - y_hat_i)^2)`
- Autocorrelation

---

## Phase 7 - Transformers, Hugging Face & Fine-Tuning LLMs
**Covers**: Chapters 14-16 (Pages 297-337)

### Topics:
1. What are Transformers? - the architecture that changed everything
2. Self-Attention mechanism
3. Multi-Head Attention
4. Positional Encoding
5. Encoder-Decoder architecture
6. Hugging Face ecosystem - pipelines, models, tokenizers
7. Common tasks: classification, NER, summarization, translation, Q&A
8. Fine-tuning LLMs with Hugging Face Trainer
9. Prompt-Tuning - soft prompts, PEFT
10. LoRA for LLMs

### Key Math Topics:
- **Attention**: `Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V`
- Query, Key, Value matrices
- Multi-head attention concatenation
- Positional encoding: `PE(pos, 2i) = sin(pos / 10000^{2i/d_model})`
- Layer normalization
- Fine-tuning vs. prompt-tuning vs. LoRA mathematics
- LoRA: `W' = W + alpha * (B * A)` where A, B are low-rank matrices

---

## Phase 8 - Generative AI: Ollama, RAG, Diffusers & LoRA
**Covers**: Chapters 17-20 (Pages 339-401)

### Topics:
1. Serving LLMs locally with Ollama
2. Building apps on top of LLMs (Python + Web)
3. What is RAG (Retrieval-Augmented Generation)?
4. Embeddings and vector similarity search
5. Building a RAG pipeline with vector databases
6. What are Diffusion Models? - the math of denoising
7. Hugging Face Diffusers library
8. Image-to-Image generation
9. Inpainting
10. Training LoRA adapters for image models
11. Fine-tuning Stable Diffusion
12. Publishing and using custom models

### Key Math Topics:
- Cosine similarity for retrieval: `cos(A,B) = (A . B) / (|A| * |B|)`
- Vector database indexing (approximate nearest neighbor)
- Forward diffusion: `q(x_t | x_{t-1}) = N(x_t; sqrt(1 - beta_t) * x_{t-1}, beta_t * I)`
- Reverse diffusion (denoising): learned by neural network
- Noise scheduling (linear, cosine)
- LoRA rank and alpha hyperparameters
- DDPM (Denoising Diffusion Probabilistic Models) loss

---

## How to Use These Notes

1. **Read the phase notes in order** - they build on each other
2. **Run the code** - every code block is meant to be executed in Google Colab or locally
3. **Work the math by hand** - at least once per concept, derive it yourself on paper
4. **After each phase**, revisit the summary and try the review questions
5. **Say "continue"** to generate the next phase

---

## Prerequisites

- Python 3.x (variables, loops, functions, classes, list comprehensions)
- Basic NumPy (array creation, indexing, broadcasting)
- High school algebra (you'll learn the rest here)
- A Google account (for Colab) or local Python + PyTorch installation

---

> **Ready to begin?** Say **"continue"** and I will generate **Phase 1: Foundations - What is ML & Getting Started with PyTorch**.
