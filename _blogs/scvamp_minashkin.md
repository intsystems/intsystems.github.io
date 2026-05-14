---
edit: true
title: "Score-Based VAMP with Fisher-Information-Based Onsager Correction"
lang: en
date: 2026-05-14
read_time: 12
authors:
  - Vladislav Minashkin
summary: "SC-VAMP: a Jacobian-free variant of Vector Approximate Message Passing that replaces expensive divergence computations with score-function norms, enabling Bayes-optimal inference with deep neural denoisers."
tags:
  - Approximate Message Passing
  - Score-based models
  - Inverse problems
  - BMM
cover: /images/blog/scvamp_minashkin/fig1.png
---

**Based on the 2026 paper by Tadashi Wadayama and Takumi Takahashi**

_If you find this topic interesting, please check out the [original paper](https://arxiv.org/abs/2601.07095)!_

## Introduction

**Vector Approximate Message Passing (VAMP)** is one of the cornerstones of modern high-dimensional statistical inference. Rooted in the statistical physics of spin glasses, VAMP and its predecessor AMP provide iterative algorithms for solving linear inverse problems of the form $\mathbf{y} = \mathbf{A}\mathbf{x}_0 + \mathbf{w}$. Their key feature is the **Onsager correction** — a seemingly mysterious term that removes harmful correlations between iterations and ensures that the algorithm's dynamics can be exactly tracked by **State Evolution (SE)** in the large-system limit. When the priors and likelihoods are Gaussian, VAMP achieves Bayes-optimal performance.

However, classical VAMP faces a critical bottleneck when we try to apply it to real-world problems. The Onsager correction requires computing the **divergence** (trace of the Jacobian) of the denoiser $\eta_t(\cdot)$ at every iteration:
$$\alpha_t = \frac{1}{N}\operatorname{div}(\eta_t) = \frac{1}{N}\sum_{i=1}^N \frac{\partial \eta_{t,i}}{\partial r_i}.$$

For simple denoisers (e.g., soft-thresholding) this is trivial. But for **deep neural network denoisers** — the kind that actually work well on natural images, medical imaging, or scientific data — computing this divergence via automatic differentiation is prohibitively expensive. It costs $\mathcal{O}(N^2)$ operations or requires memory-hungry Monte Carlo approximations, effectively negating the computational advantages of message passing.

In their 2026 paper, Wadayama and Takahashi propose **Score-based VAMP (SC-VAMP)**, a elegant reformulation that eliminates the Jacobian entirely. The key insight is that both the optimal denoiser *and* its Onsager correction can be expressed purely in terms of the **score function** $\nabla \log p(\mathbf{r})$ and the **Fisher information** $\mathbb{E}[\|\nabla \log p(\mathbf{r})\|^2]$. Since score functions are exactly what modern diffusion models learn, SC-VAMP turns any pre-trained score network into a Bayes-optimal inverse problem solver — without ever computing a derivative of the network.

## Background: From AMP to VAMP

**AMP** (Approximate Message Passing) is an iterative algorithm for linear inverse problems that alternates between a linear step and a nonlinear denoising step, with the Onsager correction subtracting the "self-interaction" of the iterate. Its asymptotic dynamics are exactly characterized by SE, a scalar recursion tracking the mean-square error.

**VAMP** generalizes AMP to handle matrices $\mathbf{A}$ of arbitrary singular value distributions (not just i.i.d. sub-Gaussian ones). It does so by using the SVD of $\mathbf{A}$ and maintaining two modules:
- **Module A (LMMSE):** processes the linear observations $\mathbf{y} = \mathbf{A}\mathbf{x} + \mathbf{w}$.
- **Module B (Denoiser):** applies a nonlinear estimator $\eta(\mathbf{r})$ to remove noise.

The Achilles' heel is the divergence computation in Module B. For a neural denoiser with millions of parameters, $\operatorname{div}(\eta)$ is a nightmare.

## Method: Score-Based VAMP

SC-VAMP resolves this by re-parameterizing the entire algorithm in terms of the score function. The paper makes three interconnected contributions.

### Tweedie's Formula and the Score Function

Suppose we observe a noisy vector $\mathbf{r} = \mathbf{x} + \gamma^{-1/2}\mathbf{z}$ where $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$. The optimal MMSE denoiser is given by **Tweedie's formula**:
$$\hat{\mathbf{x}} = \mathbb{E}[\mathbf{x}|\mathbf{r}] = \mathbf{r} + \frac{1}{\gamma}\nabla_{\mathbf{r}} \log p(\mathbf{r}; \gamma).$$

The gradient term is the **score function**. Modern diffusion models train neural networks $\mathbf{s}_\theta(\mathbf{r}, \gamma) \approx \nabla_{\mathbf{r}} \log p(\mathbf{r}; \gamma)$. Thus, we can implement the optimal denoiser using only a forward pass through a score network:
$$\mathbf{x}_{\text{post}} = \mathbf{r} + v_{\text{in}} \mathbf{s}_\theta(\mathbf{r}),$$
where $v_{\text{in}} = \gamma^{-1}$ is the input variance.

### Jacobian-Free Onsager Correction via Fisher Information

Here is the theoretical centerpiece of the paper. The authors prove that the Onsager coefficient $\alpha(v_{\text{in}})$, which normally requires the divergence of the denoiser, can be computed directly from the **conditional Fisher information**:
$$\alpha(v_{\text{in}}) = 1 - \frac{v_{\text{in}}}{N} J(\gamma),$$
where
$$J(\gamma) = \mathbb{E}_{\mathbf{r}}\left[\left\|\nabla_{\mathbf{r}} \log p(\mathbf{r}; \gamma)\right\|^2\right] = \mathbb{E}_{\mathbf{r}}\left[\left\|\mathbf{s}_\theta(\mathbf{r}, \gamma)\right\|^2\right].$$

In other words: the Onsager correction is determined by the expected squared norm of the score function. No Jacobians. No backpropagation through the denoiser. Just evaluate the score network on a mini-batch, average the squared $\ell_2$-norms, and plug into the formula.

The paper provides multiple derivations of this identity — via **Stein's identity**, via the **I-MMSE relationship** (de Bruijn's identity), and via the information-geometric interpretation of the Onsager term as a "curvature correction" governed by the local Fisher information.

### The SC-VAMP Algorithm

Putting it together, one iteration of SC-VAMP looks like this:

1. **LMMSE step (Module A):** Update using linear measurements (standard VAMP).
2. **Denoising step (Module B):**
   $$\mathbf{x}_{1,t} = \mathbf{r}_{1,t} + v_{1,t} \mathbf{s}_\theta(\mathbf{r}_{1,t}, v_{1,t}).$$
3. **Onsager correction:**
   $$\alpha_{1,t} = 1 - \frac{v_{1,t}}{N} \hat{J}_\theta, \quad \hat{J}_\theta = \frac{1}{B}\sum_{i=1}^B \|\mathbf{s}_\theta(\mathbf{r}_{1,t}^{(i)})\|^2.$$
4. **Extrinsic output:**
   $$\mathbf{r}_{2,t} = \frac{\mathbf{x}_{1,t} - \alpha_{1,t}\mathbf{r}_{1,t}}{1 - \alpha_{1,t}}.$$

The algorithm is **Jacobian-free**: the only operation involving the neural network is a forward pass to get the score. This reduces the per-iteration cost by a factor of **10–50×** compared to standard VAMP with AutoDiff-based divergence computation.

### Using Pre-Trained Denoisers

A practical bonus: if you already have a state-of-the-art denoiser $\eta_{\text{opt}}(\cdot)$ (e.g., DnCNN, DRUNet) but no explicit score model, you can extract an *implicit score* via Tweedie's formula:
$$\hat{\mathbf{s}}(\mathbf{r}) = \frac{\eta_{\text{opt}}(\mathbf{r}) - \mathbf{r}}{v_{\text{in}}}.$$

Substituting this into the Fisher information estimator yields a fully plug-and-play Onsager correction, letting SC-VAMP leverage existing denoiser libraries without retraining.

## Theoretical Guarantees

The paper demonstrates that SC-VAMP is not just a computational hack — it is theoretically sound.

### Optimality in Scalar Gaussian Channels

**Theorem 1** shows that in the classical linear Gaussian setting, SC-VAMP reduces exactly to standard Bayes-optimal VAMP. For a scalar channel $Y = X + Z$ with $X \sim \mathcal{N}(0, P)$ and $Z \sim \mathcal{N}(0, \sigma^2)$, the SE fixed point of SC-VAMP achieves the mutual information:
$$I_{\text{VAMP}} = I(X;Y) = \frac{1}{2}\log\left(1 + \frac{P}{\sigma^2}\right).$$

Moreover, the point estimate converges to the Wiener filter $\hat{x} = \frac{P}{P+\sigma^2}y$. Thus, SC-VAMP preserves the optimality of VAMP exactly where VAMP is already optimal.

### State Evolution and Decoupling

The authors verify empirically that the MSE trajectory of SC-VAMP follows the theoretical SE prediction precisely:

![](images/blog/scvamp_minashkin/fig1.png)
***Figure 1:** MSE convergence of SC-VAMP (blue) versus State Evolution theory (red dashed). The trajectories are indistinguishable, confirming that the Fisher-information-based Onsager correction is unbiased.*

This confirms that the score-norm approximation does not break the decoupling principle. The algorithm still decomposes the high-dimensional problem into independent scalar Gaussian channels in the large-system limit.

### Information-Theoretic Perspective

Perhaps the most conceptually rich part of the paper is the connection to the **entropic Central Limit Theorem**. The authors interpret the linear mixing step in VAMP as a "Gaussianizer": each iteration reduces the non-Gaussianity (in KL-divergence) of the estimation error. This provides an information-theoretic justification for why the Gaussian approximation underlying SE remains valid even beyond idealized i.i.d. settings, including nonlinear regimes.

## Experiments

The experimental section focuses on a linear observation system with a Bernoulli-Gaussian prior ($N=2000$, $M=1000$, $\rho=0.1$, SNR = 20 dB). The score function is learned via denoising score matching (DSM).

### MSE and Convergence

As shown in Figure 1, SC-VAMP tracks the theoretical SE curve almost perfectly. The algorithm converges to the same fixed point as classical VAMP with exact divergence, but with drastically lower computational cost.

### EXIT Chart Analysis

![](images/blog/scvamp_minashkin/fig2.png)
***Figure 2:** EXIT-style analysis showing Module A (observation) and Module B (denoiser) transfer characteristics. The SE trajectory (green) and actual SC-VAMP trajectory (gray dashed) follow the characteristic curves and converge to the same fixed point.*

The EXIT chart confirms that the score-based SISO modules correctly implement the MMSE estimator and that the mini-batch Fisher information estimator provides an accurate Onsager term.

## Extensions and Future Directions

The paper sketches several promising extensions:

- **Random orthogonal/unitary mixing:** To handle structured or correlated sensing matrices (where standard VAMP/AMP often fails), SC-VAMP can be combined with random rotations that "whiten" the problem.
- **Nonlinear observations:** The score-based formalism extends naturally to $\mathbf{y} = f(\mathbf{x}) + \mathbf{w}$ with deterministic nonlinearities $f$, such as sensor saturation or optical systems.
- **Flow matching:** The authors note that since score functions and velocity fields are algebraically equivalent under a given probability path, an SC-VAMP-like algorithm could be built using rectified flow or flow matching networks — potentially offering more stable training in low-noise regimes.

## Conclusion

SC-VAMP represents a significant step in unifying classical statistical-physics-based inference with modern deep learning. By reformulating the denoiser and its Onsager correction entirely through the lens of score functions and Fisher information, it eliminates the Jacobian bottleneck that has long plagued neural AMP/VAMP methods.

**Main strengths of SC-VAMP:**
- **Jacobian-free:** The Onsager correction requires only the squared norm of the score network output, enabling 10–50× speedup per iteration.
- **Plug-and-play:** Works with any pre-trained score model or denoiser; no retraining or architectural constraints.
- **Theoretically grounded:** Recovers exact Bayes-optimal VAMP in Gaussian settings and tracks State Evolution perfectly.
- **Universal:** Extends to nonlinear observations, complex priors, and structured sensing matrices.

**Limitations:**
- **Decoupling assumption:** Like all AMP/VAMP variants, SC-VAMP relies on asymptotic statistical decoupling; for highly structured finite-dimensional problems, the Gaussian approximation may be less accurate.
- **Score estimation quality:** The method inherits any bias or variance from the learned score network. In low-noise regimes, score matching can be unstable (though flow-matching extensions may remedy this).
- **Mini-batch variance:** The Fisher information is estimated via Monte Carlo; very small batch sizes could introduce variance into the Onsager term.

SC-VAMP is a rare example of a method that is simultaneously cheaper, more general, and theoretically cleaner than its predecessor. It opens the door to applying message-passing algorithms to the complex, black-box inference problems that arise in computational imaging, scientific computing, and beyond.
