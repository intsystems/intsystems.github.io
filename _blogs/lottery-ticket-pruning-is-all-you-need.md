---
edit: true
title: "Proving the Lottery Ticket Hypothesis: Pruning is All You Need"
lang: en
date: 2026-05-14
read_time: 8
authors:
  - Gleb Karpeev
summary: "Malach et al. (2020) prove a strong form of the Lottery Ticket Hypothesis: every sufficiently overparameterized randomly-initialized neural network contains a subnetwork that, without any training, approximates any target network of a given size. Training can, in principle, be replaced entirely by pruning."
tags:
  - bmm
  - Deep Learning
  - pruning
  - lottery ticket hypothesis
cover: /images/blog/lottery-ticket-pruning/lottery_figure1.png
---

## Background: from winning tickets to a stronger claim

In 2018, Frankle and Carbin introduced the [Lottery Ticket Hypothesis (LTH)](https://arxiv.org/abs/1803.03635): a dense, randomly-initialized neural network contains a sparse subnetwork, a *winning ticket*, that, when trained in isolation from the original initialization, matches the accuracy of the full network. Empirically, such tickets can be 10 to 20 times smaller than the parent network.

LTH was a striking observation, but it left a theoretical gap. *Why* do such tickets exist? Are they an artifact of optimization, or a property of random networks themselves?

In [Proving the Lottery Ticket Hypothesis: Pruning is All You Need](http://proceedings.mlr.press/v119/malach20a/malach20a.pdf) (ICML 2020), Malach, Yehudai, Shalev-Shwartz, and Shamir prove a much stronger version of LTH. Their result, sometimes called the Strong Lottery Ticket Hypothesis, says:

> A sufficiently overparameterized randomly-initialized network contains a subnetwork that approximates any target function, with no training at all. Pruning alone is enough.

This post walks through the statement, the proof idea, and what it does (and does not) say about deep learning in practice.

## The setup

Fix a target ReLU network $f^*$ of depth $\ell$ and width $n$, with bounded weights $\lVert w \rVert \le 1$. We want to approximate $f^*$ within error $\varepsilon$ on the unit ball.

The construction is:

1. Take a randomly-initialized ReLU network $G$ of depth $2\ell$ and width polynomial in $n, d, 1/\varepsilon, \log(1/\delta)$.
2. Do not train it. Instead, select a binary mask $M$ over its weights.
3. Show that $M \odot G$ approximates $f^*$ uniformly with high probability $1 - \delta$.

The notable points: depth only doubles, width is polynomial, weights are never modified; they are only kept or zeroed out.

## The main theorem (informal)

Let $f^*$ be any ReLU network of depth $\ell$, width $n$, and bounded weights. For any $\varepsilon, \delta > 0$, a random ReLU network of depth $2\ell$ and width polynomial in $n, \ell, d, 1/\varepsilon, \log(1/\delta)$ contains, with probability at least $1 - \delta$, a subnetwork that uniformly $\varepsilon$-approximates $f^*$ on the unit ball. The number of active (non-pruned) weights in the subnetwork is $O(dn + n^2 \ell)$, the same order as the parameter count of $f^*$.

In other words: expressivity of pruning a random net equals expressivity of training a net of the same target size, up to polynomial overhead in width.

The paper focuses on weight-pruning, where individual weights can be zeroed independently. A separate result (Theorem 3.2) shows that the strictly weaker model of *neuron-pruning*, in which only entire neurons can be removed, is equivalent to random features and therefore cannot achieve the same expressive power. This post discusses only the weight-pruning result.

## The proof idea: approximate each weight by two random ReLUs

The core trick is local: replace each scalar weight $w \in [-1, 1]$ in the target network by a tiny gadget built from random neurons in the random network.

### Step 1: a single weight from two ReLUs

The building block is the identity
$$
a = \sigma(a) - \sigma(-a)
$$
where $\sigma$ is the ReLU. Applied coordinate-wise, this lets us write any signed product $w \cdot x$ as
$$
w \cdot x = \sigma(w \cdot x) - \sigma(-w \cdot x)
$$
so two ReLU units with input weights $+w$ and $-w$ (and output signs $+1$ and $-1$) reproduce the linear function $x \mapsto w \cdot x$ exactly.

In the random network we do not have $\pm w$ available, but we have a *pool* of many random scalars. The pool is wide enough that, with high probability, it contains a pair of values close to $+w$ and $-w$ within accuracy $\varepsilon'$. Pruning everything else leaves a two-neuron gadget that approximates $w \cdot x$.

### Step 2: pick by pruning

The random network has many neurons in each layer. For each target weight, the construction prunes away all neurons in the pool except the two chosen ones approximating $+w$ and $-w$. Combined with the next layer's $+1/-1$ outgoing weights (also obtained by pruning), the remaining two neurons implement an approximation of $w \cdot x$.

### Step 3: error accumulation

Each weight is approximated to error $\varepsilon'$. The approximation errors propagate through $\ell$ layers, and by Lipschitz arguments on bounded-weight networks, the total error is $\mathrm{poly}(\ell, n) \cdot \varepsilon'$. Choosing $\varepsilon'$ small enough gives uniform $\varepsilon$-approximation of $f^*$.

### Why depth doubles

Each target layer is implemented by two layers of the random network: one for the pool of random scalars feeding the ReLU pair, one for combining them with $\pm 1$ outgoing weights. Hence depth $2\ell$.

### Why width is polynomial

For each target weight, the pool has to be large enough that two random scalars fall within $\varepsilon'$ of any target value in $[-1, 1]$. A covering argument on $[-1, 1]$ shows this requires $\widetilde{O}(1/\varepsilon'^2)$ random samples per weight, and a union bound over all $n^2 \ell$ weights yields a polynomial width.

## What this changes about how we think about pruning

Pruning is at least as expressive as training. Up to polynomial overhead in width, anything a trained network of size $n$ can represent can also be represented by pruning a random network of size $\mathrm{poly}(n)$. Empirical pruning algorithms are searching over a hypothesis class that is, in principle, rich enough.

The winning ticket is universal. The same random network contains tickets for *every* target; the choice of mask depends on $f^*$, but the random weights do not.

It partially explains why pruning works so well after training. If subnetworks of the *initial* random network can already approximate good functions, it is not surprising that subnetworks of a *trained* network can match its accuracy.

## Important caveats

The result is an existence theorem, and several gaps remain between the proof and practice:

1. No algorithm. The proof tells us a good mask exists but not how to find it. Finding the optimal subnetwork is, in the worst case, computationally hard. Practical methods (magnitude pruning, edge-popup, lottery-ticket rewinding) are heuristics.
2. Polynomial overparameterization. The width bound is polynomial in the target size, the input dimension, and $1/\varepsilon$. The exponents in the original proof are not tight, and for realistic $n$ the implied constants are large.
3. Bounded weights and bounded inputs. The construction requires weight and input norms to be bounded, which is the standard but non-trivial assumption.
4. No generalization claim. The theorem is about *approximation*, not learning. It does not say that a pruned random network will generalize from finite data the way a trained one does.

Follow-up work has tightened these bounds. [Pensia et al. (2020)](https://arxiv.org/abs/2006.07990) showed that the width overhead can be reduced to *logarithmic* in $1/\varepsilon$, bringing the strong LTH much closer to a tight statement.

## Takeaways

Frankle and Carbin's empirical LTH said: train, then prune to find a winning ticket. Malach et al.'s strong LTH says: you don't even need to train. The winning ticket is already there in the random initialization, waiting to be uncovered by a mask.

The proof is a clean, constructive argument: approximate every target weight by a difference of two random ReLUs, prune away the rest, and propagate the error through the layers.

The result is theoretical, and finding the mask remains the hard part, but it cleanly separates expressivity (which pruning a random net already has) from the algorithmic question of how to find a good subnetwork.

Pruning, viewed this way, is not a compression heuristic applied after the fact. It is a legitimate alternative form of "training", and the strong LTH is the formal statement that it is, in principle, sufficient.

## References

1. Malach, E., Yehudai, G., Shalev-Shwartz, S., & Shamir, O. (2020). *Proving the Lottery Ticket Hypothesis: Pruning is All You Need*. ICML 2020. [PDF](http://proceedings.mlr.press/v119/malach20a/malach20a.pdf)
2. Frankle, J., & Carbin, M. (2018). *The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks*. arXiv preprint, later published at ICLR 2019. [arXiv:1803.03635](https://arxiv.org/abs/1803.03635)
3. Pensia, A., Rajput, S., Nagle, A., Vishwakarma, H., & Papailiopoulos, D. (2020). *Optimal Lottery Tickets via Subset Sum: Logarithmic Over-Parameterization is Sufficient*. NeurIPS 2020. [arXiv:2006.07990](https://arxiv.org/abs/2006.07990)
4. Ramanujan, V., Wortsman, M., Kembhavi, A., Farhadi, A., & Rastegari, M. (2020). *What's Hidden in a Randomly Weighted Neural Network?* CVPR 2020. [arXiv:1911.13299](https://arxiv.org/abs/1911.13299)
