# Simple RNN — In-Depth Intuition

This README builds intuition for a vanilla Recurrent Neural Network (RNN), from processing words sequentially to training with Backpropagation Through Time (BPTT), and finally understanding why vanilla RNNs can struggle with long-term dependencies.

## Table of Contents

1. [Introduction](#1-introduction)
2. [Preparing Text for an RNN](#2-preparing-text-for-an-rnn)
3. [RNN Architecture and Forward Propagation](#3-rnn-architecture-and-forward-propagation)
4. [Worked Forward-Pass Example](#4-worked-forward-pass-example)
5. [Calculating the Loss](#5-calculating-the-loss)
6. [Backpropagation Through Time (BPTT)](#6-backpropagation-through-time-bptt)
7. [A Small Gradient Example](#7-a-small-gradient-example)
8. [Training Loop](#8-training-loop)
9. [Vanishing and Exploding Gradients](#9-vanishing-and-exploding-gradients)
10. [How LSTMs and GRUs Help](#10-how-lstms-and-grus-help)
11. [Common Misconceptions](#11-common-misconceptions)
12. [Summary and Formula Sheet](#12-summary-and-formula-sheet)

---

## 1. Introduction

A Recurrent Neural Network (RNN) is a neural network designed for sequential data, such as text, time series, and speech.

A feedforward network processes an input without an explicit recurrent state. An RNN maintains a **hidden state**, which carries information from earlier time steps into the current computation.

Consider this sentiment-classification dataset:

| Text                 | Output |
| -------------------- | -----: |
| The food is good     |      1 |
| The food is bad      |      0 |
| The food is not good |      0 |

For this example, `1` means positive sentiment and `0` means negative sentiment.

The phrase `not good` illustrates why context matters: the word `good` is usually positive, but negation changes the meaning of the phrase.

Our goal is to process each sentence one word at a time and predict its sentiment.

## 2. Preparing Text for an RNN

### 2.1 Vocabulary and tokenization

A vocabulary contains the unique tokens used by the model. For the dataset above, one possible vocabulary is:

```text
[the, food, is, good, bad, not]
```

Tokenization converts a sentence into tokens:

```text
"The food is not good"
        ↓
["the", "food", "is", "not", "good"]
```

Real preprocessing often also handles punctuation, capitalization, unknown words, and padding for batches of different-length sequences.

### 2.2 Convert words into vectors

Neural networks operate on numbers, so each token must be represented numerically.

**One-hot encoding** represents each word using a vector whose length equals the vocabulary size. For the vocabulary `[the, food, is, good, bad, not]`:

```text
the  = [1, 0, 0, 0, 0, 0]
food = [0, 1, 0, 0, 0, 0]
is   = [0, 0, 1, 0, 0, 0]
good = [0, 0, 0, 1, 0, 0]
bad  = [0, 0, 0, 0, 1, 0]
not  = [0, 0, 0, 0, 0, 1]
```

One-hot vectors are easy to understand, but they are sparse and do not directly encode similarity between words.

An **embedding** maps each token to a dense vector, such as a vector with 50, 100, or 300 values. Embeddings may be pretrained or learned during model training.

We will use `x_t` to denote the vector for the word at time step `t`.

### 2.3 Input and hidden dimensions

Suppose each word vector has 6 dimensions and the RNN has 3 hidden units.

Then:

- `x_t` has shape `(6,)`.
- `h_t` has shape `(3,)`.
- `W_xh` has shape `(3, 6)`.
- `W_hh` has shape `(3, 3)`.
- `b_h` has shape `(3,)`.

The number of hidden units does **not** have to equal the input-vector dimension.

For a batch of examples, an implementation usually adds a batch dimension; the shapes above describe one sequence at one time step.

## 3. RNN Architecture and Forward Propagation

### 3.1 The recurrent idea

For the sentence:

```text
The food is good
```

the RNN processes:

```text
x₁ = embedding("the")
x₂ = embedding("food")
x₃ = embedding("is")
x₄ = embedding("good")
```

At each step, the RNN uses both the current input and the previous hidden state:

```text
x₁ ──► [RNN cell] ──► h₁
                         │
x₂ ──► [RNN cell] ◄──────┘ ──► h₂
                                  │
x₃ ──► [RNN cell] ◄───────────────┘ ──► h₃
                                           │
x₄ ──► [RNN cell] ◄────────────────────────┘ ──► h₄
```

This is a conceptual illustration. When an RNN is drawn “unrolled,” each cell represents the same computation and shares the same parameters.

The hidden state is a learned summary of the sequence prefix seen so far. It is not a perfect record of every previous word.

### 3.2 Hidden-state equations

Define the pre-activation:

\[
a*t = W*{xh}x*t + W*{hh}h\_{t-1} + b_h
\]

Then calculate the hidden state:

\[
h_t = \phi(a_t)
\]

Where:

- `x_t`: current input vector.
- `h_(t-1)`: previous hidden state.
- `h_t`: current hidden state.
- `W_xh`: input-to-hidden weights.
- `W_hh`: hidden-to-hidden recurrent weights.
- `b_h`: hidden bias.
- `φ`: hidden activation function, often `tanh`.

A common initialization is:

\[
h_0 = \mathbf{0}
\]

For four words:

\[
h*1 = \phi(W*{xh}x*1 + W*{hh}h_0 + b_h)
\]

\[
h*2 = \phi(W*{xh}x*2 + W*{hh}h_1 + b_h)
\]

\[
h*3 = \phi(W*{xh}x*3 + W*{hh}h_2 + b_h)
\]

\[
h*4 = \phi(W*{xh}x*4 + W*{hh}h_3 + b_h)
\]

**Important:** `W_xh`, `W_hh`, and `b_h` are shared across all time steps. The model does not normally learn a separate recurrent weight matrix for each word position.

### 3.3 Why does the hidden state provide context?

After reading `the`, the hidden state is `h₁`. After reading `the food`, it becomes `h₂`. After reading `the food is`, it becomes `h₃`.

The next hidden state is computed from the previous one, so information from earlier words can influence later computations.

However, the vanilla RNN compresses the history into a fixed-size vector. It can lose information, particularly when the sequence is long.

### 3.4 Producing a sentence-level prediction

For sentiment classification, we can use the final hidden state `h₄` to predict the sentiment.

Calculate the output logit:

\[
z = W\_{hy}h_4 + b_y
\]

For binary classification, apply the sigmoid function:

\[
\hat y = \sigma(z)=\frac{1}{1+e^{-z}}
\]

Here:

- `W_hy` is the hidden-to-output weight matrix.
- `b_y` is the output bias.
- `z` is the output logit.
- `ŷ` is the predicted probability of class 1.

For example, if `ŷ = 0.85`, the model predicts an 85% probability of positive sentiment. With a threshold of `0.5`, this would be classified as positive.

This README uses a **many-to-one RNN**: it reads many input tokens and produces one prediction for the entire sequence. Language generation often uses a different setup, predicting an output at each time step.

![RNN forward propagation](../course_images/image_12_1.png)

## 4. Worked Forward-Pass Example

The following toy example uses one hidden unit, one scalar input per time step, and simple numbers so the calculations are easy to follow. These values are illustrative, not trained parameters.

Let:

```text
x₁ = 1
x₂ = 2
x₃ = 1

W_xh = 0.5
W_hh = 0.1
b_h  = 0
h₀   = 0
```

Use `tanh` as the hidden activation:

\[
h*t=\tanh(W*{xh}x*t+W*{hh}h\_{t-1}+b_h)
\]

At time step 1:

\[
h_1=\tanh(0.5(1)+0.1(0))=\tanh(0.5)\approx0.4621
\]

At time step 2:

\[
h_2=\tanh(0.5(2)+0.1(0.4621))
\]

\[
h_2=\tanh(1.0462)\approx0.7802
\]

At time step 3:

\[
h_3=\tanh(0.5(1)+0.1(0.7802))
\]

\[
h_3=\tanh(0.5780)\approx0.5213
\]

Notice that `h₂` depends on both `x₂` and `h₁`. That is how the earlier input can affect later states.

For a final prediction, suppose:

```text
W_hy = 2
b_y  = 0
```

Then:

\[
z=2h_3\approx1.0426
\]

\[
\hat y=\sigma(1.0426)\approx0.7393
\]

The model's prediction is approximately `0.739`. It would classify this as class 1 at a threshold of `0.5`. Again, this example demonstrates the arithmetic; it does not establish that the model has learned useful sentiment features.

## 5. Calculating the Loss

The loss measures how far the prediction is from the target. For binary classification, a common choice is **binary cross-entropy (BCE)**:

\[
L=-[y\log(\hat y)+(1-y)\log(1-\hat y)]
\]

Where:

- `y` is the actual label (`0` or `1`).
- `ŷ` is the predicted probability.
- `L` is the loss for one example.

### Example

Suppose the actual label is `y = 1` and the predicted probability is `ŷ = 0.85`.

\[
L=-\log(0.85)\approx0.1625
\]

If the same positive example received a prediction of `0.10`, its loss would be:

\[
L=-\log(0.10)\approx2.3026
\]

The second prediction is much worse because the model assigned a low probability to the correct class.

**Important distinction:** `y - ŷ` is not the loss. With sigmoid output and binary cross-entropy, the derivative of the loss with respect to the output logit `z` is:

\[
\frac{\partial L}{\partial z}=\hat y-y
\]

We use this derivative when calculating the output-layer gradients.

## 6. Backpropagation Through Time (BPTT)

### 6.1 Why do we need backpropagation?

Forward propagation computes the prediction. Training requires us to adjust the model parameters so that the loss generally decreases.

Gradient descent updates a parameter using:

\[
W*{\text{new}}=W*{\text{old}}-\eta\frac{\partial L}{\partial W}
\]

Where:

- `W_old` is the current parameter.
- `W_new` is the updated parameter.
- `η` is the learning rate.
- `∂L/∂W` is the gradient of the loss with respect to that parameter.

The learning rate controls the step size. Too large a learning rate can make training unstable; too small a learning rate can make it slow.

### 6.2 Why is an RNN gradient different?

In a feedforward network, the computation graph flows through layers. In an RNN, each hidden state also depends on the previous hidden state.

For example:

```text
x₁ → h₁ → h₂ → h₃ → h₄ → ŷ → Loss
      ↑     ↑     ↑     ↑
     x₁    x₂    x₃    x₄
```

The same recurrent weights `W_hh` participate at every step. A change to those weights affects the hidden state at each position and can therefore affect the final prediction through multiple paths.

**Backpropagation Through Time (BPTT)** calculates these gradients by treating the recurrent computation as an unrolled sequence and applying the chain rule backward through time.

### 6.3 Parameters to update

For the vanilla RNN described here, the trainable parameters are:

- `W_xh`: input-to-hidden weights.
- `W_hh`: hidden-to-hidden recurrent weights.
- `b_h`: hidden bias.
- `W_hy`: hidden-to-output weights.
- `b_y`: output bias.

We will assume a single loss calculated from the final hidden state. If the model produces a loss at every time step, the equations must include the additional output-gradient terms.

### 6.4 Step 1: Calculate the output gradient

Recall:

\[
z=W\_{hy}h_T+b_y
\]

\[
\hat y=\sigma(z)
\]

For sigmoid combined with binary cross-entropy:

\[
\delta_y=\frac{\partial L}{\partial z}=\hat y-y
\]

The output-weight gradient is:

\[
\frac{\partial L}{\partial W\_{hy}}=\delta_y h_T^T
\]

The output-bias gradient is:

\[
\frac{\partial L}{\partial b_y}=\delta_y
\]

Here, `T` denotes the final time step, while the superscript `T` on a vector or matrix denotes transpose. The distinction should be clear from the context.

### 6.5 Step 2: Calculate the final hidden-state gradient

Recall:

\[
a*T=W*{xh}x*T+W*{hh}h\_{T-1}+b_h
\]

\[
h_T=\phi(a_T)
\]

First, propagate the output gradient into the final hidden state:

\[
\frac{\partial L}{\partial h*T}=W*{hy}^T\delta_y
\]

Define the pre-activation gradient:

\[
\delta_T=\frac{\partial L}{\partial a_T}
\]

Using the chain rule:

\[
\delta*T=(W*{hy}^T\delta_y)\odot\phi'(a_T)
\]

The symbol `⊙` means element-wise multiplication.

If `φ` is `tanh`, its derivative can be written in terms of the hidden state:

\[
\phi'(a_t)=1-h_t^2
\]

Therefore:

\[
\delta*T=(W*{hy}^T\delta_y)\odot(1-h_T^2)
\]

### 6.6 Step 3: Propagate the gradient backward through time

For earlier time steps, the hidden-state gradient follows the recurrence:

\[
\delta*t=(W*{hh}^T\delta\_{t+1})\odot\phi'(a_t)
\]

For a sequence of four time steps:

\[
\delta*4=(W*{hy}^T\delta_y)\odot\phi'(a_4)
\]

\[
\delta*3=(W*{hh}^T\delta_4)\odot\phi'(a_3)
\]

\[
\delta*2=(W*{hh}^T\delta_3)\odot\phi'(a_2)
\]

\[
\delta*1=(W*{hh}^T\delta_2)\odot\phi'(a_1)
\]

The gradient at an earlier time step includes the effect of that step on all later hidden states and ultimately on the loss.

This repeated multiplication is one reason gradients may shrink or grow significantly in long sequences.

### 6.7 Step 4: Calculate the recurrent-weight gradient

The recurrent weights `W_hh` are shared across all time steps:

\[
a*t=W*{xh}x*t+W*{hh}h\_{t-1}+b_h
\]

Consequently, the total gradient for `W_hh` is the sum of the contribution at every time step:

\[
\frac{\partial L}{\partial W*{hh}}=\sum*{t=1}^{T}\delta*t h*{t-1}^T
\]

For four time steps:

\[
\frac{\partial L}{\partial W\_{hh}}=
\delta_1h_0^T+\delta_2h_1^T+\delta_3h_2^T+\delta_4h_3^T
\]

Update the recurrent weights:

\[
W*{hh}^{new}=W*{hh}^{old}-\eta\frac{\partial L}{\partial W\_{hh}}
\]

We sum these gradients because the same recurrent weight matrix is reused at each time step.

### 6.8 Step 5: Calculate the input-weight gradient

The input-to-hidden weights `W_xh` are also shared across time:

\[
\frac{\partial L}{\partial W*{xh}}=\sum*{t=1}^{T}\delta_t x_t^T
\]

For four time steps:

\[
\frac{\partial L}{\partial W\_{xh}}=
\delta_1x_1^T+\delta_2x_2^T+\delta_3x_3^T+\delta_4x_4^T
\]

Update the input weights:

\[
W*{xh}^{new}=W*{xh}^{old}-\eta\frac{\partial L}{\partial W\_{xh}}
\]

### 6.9 Step 6: Calculate the bias gradients

The hidden bias is shared across time steps, so its gradient is:

\[
\frac{\partial L}{\partial b*h}=\sum*{t=1}^{T}\delta_t
\]

Update it:

\[
b_h^{new}=b_h^{old}-\eta\frac{\partial L}{\partial b_h}
\]

The output parameters are updated similarly:

\[
W*{hy}^{new}=W*{hy}^{old}-\eta\frac{\partial L}{\partial W\_{hy}}
\]

\[
b_y^{new}=b_y^{old}-\eta\frac{\partial L}{\partial b_y}
\]

After all gradients are calculated, update the parameters and run another forward pass with the new values.

![RNN backpropagation through time](../course_images/image_12_2.jpeg)

## 7. A Small Gradient Example

Consider a scalar hidden state and a three-step sequence. This simplified example illustrates how the gradient flows backward; it is not a full training run.

Suppose the final hidden state has a gradient arriving from the output:

```text
∂L / ∂h₃ = 0.8
```

Assume the recurrent weight is:

```text
W_hh = 0.5
```

And assume the activation derivative at each step is:

```text
φ'(a₃) = 0.9
φ'(a₂) = 0.8
φ'(a₁) = 0.7
```

The final pre-activation gradient is:

\[
\delta_3=\frac{\partial L}{\partial h_3}\phi'(a_3)
=0.8(0.9)=0.72
\]

At the previous time step:

\[
\delta*2=W*{hh}\delta_3\phi'(a_2)
\]

\[
\delta_2=0.5(0.72)(0.8)=0.288
\]

At the first time step:

\[
\delta*1=W*{hh}\delta_2\phi'(a_1)
\]

\[
\delta_1=0.5(0.288)(0.7)=0.1008
\]

The gradient has decreased from `0.72` to `0.288` to `0.1008` as it propagates backward. This demonstrates how repeated multiplication by factors smaller than one can shrink a gradient.

In a real RNN, the recurrent weight is a matrix, the activation derivatives are vectors, and the gradient calculations use matrix multiplication and element-wise multiplication.

## 8. Complete RNN Training Loop

The overall training process is:

1. Convert each sentence into a sequence of vectors.
2. Initialize the hidden state.
3. Process the input vectors in order.
4. Calculate the final prediction.
5. Calculate the loss against the target.
6. Run BPTT to calculate gradients.
7. Update all trainable parameters using gradient descent or an optimizer such as Adam.
8. Repeat for subsequent training examples and epochs.

The goal is to minimize the training objective. Gradient descent does not guarantee that training will reach a global minimum.

## 9. Vanishing and Exploding Gradients

Vanilla RNNs can struggle to learn dependencies across long sequences because gradients are repeatedly propagated through recurrent connections and activation functions.

### 9.1 Vanishing gradients

A gradient can become progressively smaller as it is propagated backward through many time steps.

For example, if a simplified gradient is multiplied by `0.5` at each step:

```text
1 step:   0.5
2 steps:  0.25
3 steps:  0.125
4 steps:  0.0625
...
```

After many steps, the gradient may become so small that the early time steps receive almost no useful learning signal.

Consequences:

- The model struggles to learn long-term dependencies.
- Earlier words may have little influence on what the model learns.
- Parameters associated with early sequence positions may receive very small updates.

This does **not** mean earlier words can never influence predictions. It means the learning signal used to train those long-range dependencies may become weak.

### 9.2 Exploding gradients

The opposite can happen when repeated multiplication causes gradient magnitudes to grow:

```text
1 step:   2
2 steps:  4
3 steps:  8
4 steps:  16
...
```

Very large gradients can produce excessively large parameter updates, unstable training, or numerical overflow.

### 9.3 How can we address these problems?

Common approaches include:

1. **Gradient clipping:** Limit the gradient norm to reduce the risk of exploding gradients.
2. **LSTM:** Uses a gated memory mechanism that helps preserve information over longer sequences.
3. **GRU:** Uses update and reset gates to control information flow.
4. **Careful initialization and optimization:** Appropriate weight initialization and learning rates can improve training stability.
5. **ReLU or Leaky ReLU:** These may help with some gradient-flow issues, but they do not reliably solve the vanishing-gradient problem in vanilla RNNs. ReLU can also contribute to exploding activations or inactive units.

For a basic RNN learning project, gradient clipping is a useful technique to understand, while LSTMs and GRUs are important architectural alternatives.

## 10. How LSTMs and GRUs Help

### LSTM (Long Short-Term Memory)

An LSTM adds a cell state and gates that control what information is retained, added, or removed.

The cell state provides a path through which information and gradients can often travel more effectively across many time steps.

### GRU (Gated Recurrent Unit)

A GRU uses update and reset gates to control the flow of information. It has a simpler state structure than an LSTM and can be effective on many sequence-learning tasks.

Neither architecture guarantees perfect long-term memory, but both were designed to make learning longer dependencies easier than with a basic vanilla RNN.

## 11. Common Misconceptions

### Misconception 1: Every time step has different recurrent weights

**Correction:** The same recurrent weights are reused across time. The total gradient for a shared parameter is accumulated across its uses.

### Misconception 2: The loss is `y - ŷ`

**Correction:** A loss function such as binary cross-entropy measures the prediction error. For sigmoid plus binary cross-entropy, the gradient with respect to the logit is `ŷ - y`.

### Misconception 3: The hidden state remembers the entire sentence perfectly

**Correction:** The hidden state is a fixed-size learned representation of the sequence prefix. Information can be lost.

### Misconception 4: ReLU completely fixes vanishing gradients

**Correction:** ReLU can help in some settings, but it does not remove the recurrent dependency or guarantee stable gradient flow. LSTMs and GRUs introduce gating mechanisms specifically to improve the learning of longer dependencies.

### Misconception 5: Gradient descent always finds the global minimum

**Correction:** Neural-network optimization is generally non-convex. Gradient-based optimizers seek to reduce the objective but do not guarantee a global minimum.

## 12. Summary and Formula Sheet

| Operation                 | Formula                               |
| ------------------------- | ------------------------------------- |
| Hidden pre-activation     | `a_t = W_xh x_t + W_hh h_(t-1) + b_h` |
| Hidden state              | `h_t = φ(a_t)`                        |
| Output logit              | `z = W_hy h_T + b_y`                  |
| Prediction                | `ŷ = σ(z)`                            |
| Binary cross-entropy      | `L = -[y log(ŷ) + (1-y) log(1-ŷ)]`    |
| Output logit gradient     | `δ_y = ŷ - y`                         |
| Final hidden gradient     | `δ_T = (W_hyᵀ δ_y) ⊙ φ'(a_T)`         |
| Earlier hidden gradient   | `δ_t = (W_hhᵀ δ_(t+1)) ⊙ φ'(a_t)`     |
| Recurrent-weight gradient | `∂L/∂W_hh = Σ_t δ_t h_(t-1)ᵀ`         |
| Input-weight gradient     | `∂L/∂W_xh = Σ_t δ_t x_tᵀ`             |
| Hidden-bias gradient      | `∂L/∂b_h = Σ_t δ_t`                   |
| Gradient descent          | `W_new = W_old - η × ∂L/∂W`           |

**Final takeaway:** Forward propagation processes the sequence and produces a prediction. BPTT calculates how the trainable parameters contributed to the loss, and gradient descent uses those gradients to update the parameters. Vanilla RNNs can struggle with long-range dependencies because gradients may vanish or explode, motivating techniques such as gradient clipping and architectures such as LSTMs and GRUs.
