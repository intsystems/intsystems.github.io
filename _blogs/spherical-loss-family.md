---
edit: true
title: "An Exploration of Softmax Alternatives Belonging to the Spherical Loss Family"
lang: en
date: 2025-12-11
read_time: 6
authors:
  - Fedor Sobolevsky
summary: "Spherical loss functions and their performance compared to log-softmax in multi-class classification tasks."
tags:
  - Loss Functions
  - BMM
cover: /images/blog/spherical-loss-family.png
---

**Based on the ICLR conference paper by Alexandre de Brébisson and Pascal Vincent (2016)**

_If you find this topic interesting, please check out the [original paper](https://arxiv.org/abs/1511.05042)_!

## Motivation

In multi-class classification problems, the standard loss function used by the overwhelming majority of machine learning **models** is the log-softmax function. It is certainly a convenient function with interpretable values and works well — but is it necessarily the best choice of loss function, period? The answer may be: not always. In this blog post, we’ll dive into research conducted by Alexandre de Brébisson and Pascal Vincent on loss functions for multi-class classification tasks and explore some alternative methods from the family of **spherical loss functions**.

But let’s take this step by step.

### The Multi-Class Classification Task

First, we’ll establish a mathematical model of a loss function. Given a neural classification model, let’s denote the output of its last hidden layer as $\mathbf{o}$, where $\mathbf{o}$ is a $d$-dimensional vector. Suppose we have $D$ classes in our classification task. Let us denote the target vector $y$ and its non-zero component’s index as $c$. Then, a **loss function** is a function of the last layer output and $c$:

$$
\mathcal{L} = \mathcal{L}(\mathbf{o}, c).
$$

### Current Approach: Log-Softmax

The **log-softmax** loss function is defined in our notation as follows:

$$
L(\mathbf{o}, c) = -\log\frac{e^{o_c}}{\sum_{k=1}^D e^{o_k}} = -o_c + \log \sum_{k=1}^D e^{o_k}.
$$

This loss function is standard in multi-class classification, but it may be a bit too computationally expensive in some cases: gradient updates require $\mathcal{O}(D\times d)$ calculations, which for large output dimensionality $D$ (i.e., in tasks with large numbers of classes, such as language modeling) may be suboptimal. Alternative loss functions, which we’ll discuss below, may provide a solution to this problem.

---

## The Spherical Loss Family

Let’s introduce a new family of loss functions depending on symmetric statistics of the hidden layer output vector $\mathbf{o}$. A loss belongs to the spherical family if it depends only on:

- $s = \sum_i o_i$,
- $q = \sum_i o_i^2 = \|\boldsymbol{o}\|^2_2$,
- $o_c$,
- $y_c$ for the target class $c$.

Then such a function can be written in the form

$$
\mathcal{L} = \mathcal{L}(s, q, o_c, y_c).
$$

This definition may sound restrictive, but this family of functions is in fact quite diverse. One simple example of a spherical loss function—though for regression rather than classification—is MSE (mean squared error).

A neat property of spherical loss functions, discovered and proven by the paper’s authors together with Guillaume Bouchard in their [2015 paper](https://proceedings.neurips.cc/paper/2015/file/7f5d04d189dfb634e6a85bb9d9adf21e-Paper.pdf), is that they allow for efficient gradient computation in $\mathcal{O}(d^2)$ instead of $\mathcal{O}(D\times d)$, which for large numbers of classes $D$ is noticeably better.

### Spherical Softmax

One simple example of a spherical loss function is the **spherical softmax loss**. It is defined as follows:

$$
L_{\text{log sph soft}} = -\log f_{\text{sph soft}}(\boldsymbol{o})_c, \quad f_{\text{sph soft}}(\boldsymbol{o})_k = \frac{o_k^2 + \varepsilon}{\sum_i (o_i^2 + \varepsilon)},
$$

where $\varepsilon$ is a small additive term for numerical stability in case $q$ is very small. It is fairly trivial to prove that this function is spherical, but it has more useful properties than just that:

- $L_{\text{log sph soft}}$ is invariant to scaling of $\boldsymbol{o}$.
- It is an even function, i.e., it ignores the sign of $o_k$.

However, it should be kept in mind that this loss function also requires careful tuning of the hyperparameter $\varepsilon$ for numerical stability.

### Taylor Softmax

Another loss function from the spherical loss family is based on the second-order Taylor decomposition of the exponent in the regular softmax loss: $\exp(x) \approx 1 + x + \frac{1}{2}x^2$. It is defined as follows:

$$
L_{\text{log tay soft}} = -\log f_{\text{tay soft}}(\boldsymbol{o})_c, \quad f_{\text{tay soft}}(\boldsymbol{o})_k = \frac{1 + o_k + \frac{1}{2}o_k^2}{\sum_i (1 + o_i + \frac{1}{2}o_i^2)}.
$$

It isn’t hard to see that this is also a spherical loss function that depends only on $s$, $q$, and $o_c$. Moreover, this function, in contrast to the previous one, doesn’t require any hyperparameters and is numerically stable. Another distinctive feature of this function is that it is slightly asymmetric around zero. Interestingly, the paper’s authors hypothesize that this is actually a positive feature, which we’ll see in action later on.

### Spherical Upper Bound for Log-Softmax

We can’t go on without mentioning one more loss function discussed in the paper, which is a spherical upper bound of the log-softmax function. It is derived from an upper bound for the log-sum of exponentials proposed by [Bouchard (2007)](https://d1wqtxts1xzle7.cloudfront.net/6050190/nips_wrkshp_subm-libre.pdf?1390844193=&response-content-disposition=inline%3B+filename%3DEfficient_bounds_for_the_softmax_functio.pdf&Expires=1765223935&Signature=gOc6yncS0Obya~5QQ9bSvCV2MBsetfiM392gytfs7hjhuiGc1ROg4EYg15zUfG~bvLb8pbEP~TTFjd8mWa0GN7zmOohyaDiCS53eAHdgUo2w9liS5Lj-WBmx4usNBV4sdhWV-cUBYaF3ZfQCpHiUgVlbgENG0VQocW8bS4I5k~Y34iaf1oMCvUVQBiL0ifDHpiIhrm9j~g6lW-zow-sAeg4x-JNtKj1xXll0APDclFNLz1AfSsuCc3ZdhZA6LjqW-zAsBNWX0NBLqdaKi0TOwAdter-ekVkiBRvMfVli0r71IlohGyfLsQiylI0dFsmfGAXIjx4mS3X9rp8HLdYW7A__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA) and has the following monstrosity of an expression:

$$
L \leq \left(-\frac{(D-2)^2}{16D}\frac{1}{\lambda(\xi)}-\frac{D}{2}\xi-D\lambda(\xi)\xi^2+\right.
$$

$$
\left.+D\log(1+e^{\xi})+\frac{1}{D}s+\left(q-\frac{s^2}{D}\right)\lambda(\xi)-o_c\right), \text{ where}
$$

$$
\lambda(\xi)=\frac{1}{2\xi}\left(\frac{1}{1+e^{-\xi}}-\frac{1}{2}\right).
$$

You can guess that I didn’t type that one out by hand. Ironically, the most complex of the functions discussed in the paper performed the worst. So poorly, in fact, that the authors mentioned it briefly at the beginning of the experiments section and then withdrew it from consideration. In our discussion of experimental results, we will follow their example.

---

## Experimental Results

To test the loss functions in action, the authors compared log-softmax and different spherical alternatives on several tasks: with low-dimensional outputs, like image classification on MNIST and CIFAR-10, and with higher-dimensional outputs, like classification on CIFAR-100 and a language modeling task on the PennTree dataset. The goal was not to reach state-of-the-art performance on each task but to compare the influence of each loss given the same classification model.

### Low-Dimensional Outputs

|         Loss          | MNIST Error | CIFAR-10 Error |
| :-------------------: | :---------: | :------------: |
|      Log-Softmax      |   0.812%    |     8.52%      |
|  Log-Taylor Softmax   | **0.785%**  |   **8.07%**    |
| Log-Spherical Softmax |   0.828%    |     8.37%      |

Experiments on datasets with low numbers of classes showed that such tasks may indeed be a scenario where spherical loss functions are superior to the usual log-softmax. Particularly, Taylor softmax noticeably outperforms log-softmax on both tasks with low-dimensional outputs, and spherical softmax also achieves comparable results.

### Higher-Dimensional Outputs

|         Loss          | CIFAR-100 Error | PennTree Perplexity |
| :-------------------: | :-------------: | :-----------------: |
|      Log-Softmax      |    **32.4%**    |      **126.7**      |
|  Log-Taylor Softmax   |      33.1%      |        147.2        |
| Log-Spherical Softmax |      33.1%      |        149.2        |

On the contrary, log-softmax performs better as output dimension increases. This is particularly prominent in language modeling tasks with huge vocabulary sizes: here the difference between log-softmax and spherical losses is _really_ noticeable. The paper’s authors suggest that this may be caused by the exponential in softmax better handling high-dimensional competition. It should be noted that such tasks with higher-dimensional outputs are the ones where the gain in computational efficiency achievable by spherical losses is most visible.

---

## Conclusion

So, what did we end up with? Spherical losses (especially Taylor softmax) can outperform log-softmax on small-output tasks (e.g., MNIST, CIFAR-10), but for high-dimensional outputs (CIFAR-100, language modeling), log-softmax remains superior. The paper we discussed thus highlights that alternatives to log-softmax are in fact worth considering, but the choice of loss function should be task-specific.

In tasks with high-dimensional outputs, spherical losses enable efficient training but may lack the discriminative power of log-softmax. They are worth considering for specific applications where efficiency is key. In tasks with low-dimensional outputs, on the other hand, the use of spherical loss functions, especially Taylor softmax, can benefit model accuracy — so in such tasks, we may have just discovered the softmax killer.
