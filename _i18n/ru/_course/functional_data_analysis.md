**Channels**

- [The 2025 course seminar](https://t.me/+XyXmEXRlrXB9dZKD)
- [The chat-link FDA group](https://t.me/+XyXmEXRlrXB9dZKD)

**When**

- September 4, 11, 18, 25 — **Thursdays** at 10:30 · [m1p.org/go_zoom](https://m1p.org/go_zoom)
- October (most likely) — **Saturdays** at 10:30

---

## Foundation models for spatial-time series

Foundation AI models are universal models for a wide set of problems. This project investigates their theoretical properties on **spatial-time series**—data used across sciences to generalize knowledge and make forecasts. Core user-level tasks: _forecasting_ and _generation_ of time series; _analysis_ and _classification_; _change-point detection_; _causal inference_. These models are trained on massive datasets. Our goal is to compare architectures to find an optimal one that solves the above for a broad range of spatial time series.

## Functional data analysis

We assume continuous time and study state-space changes $\frac{d\mathbf{x}}{dt}$ via neural ODEs/SDEs. We analyze multivariate/multidimensional series with tensor representations; model strong cross-correlations in **Riemannian** spaces. Many medical series are periodic; the base model is the pendulum $\frac{d^2 x}{dt^2} = -c\sin x$. We use physics-informed neural networks (PINNs). Practical experiments involve multiple sources; we use canonical correlation analysis with a **latent state space** to align source/target manifolds and enable generation in both.

## Applications

Any field with continuous time/space data from multimodal sources: climate, neural interfaces, solid-state physics, electronics, fluid dynamics, and more. We collect both theory and practice.

---

## Fall 2025: Foundation models for time series

### Topics to discuss

1. State Space Models, Convolution, SSA, SSM (Spectral Submanifolds)
2. Neural & Controlled ODE, Neural PDE, Geometric Learning
3. Operator Learning, Physics-informed learning, multimodeling
4. Spatio-Temporal Graph Modeling: graph convolution & metric tensors
5. Riemannian models; time series generation
6. AI for science: mathematical modeling principles

Outside the course: data-driven tensor analysis, differential forms, spinors.

### State of the Art in 2025

In December 2024, a NeurIPS workshop “Foundational models for science” reflected this theme:

1. Foundation Models for Science: Progress, Opportunities, and Challenges — [URL](https://neurips.cc/virtual/2024/workshop/84714)
2. Foundation Models for the Earth system — [UPL, no paper](https://neurips.cc/virtual/2024/107817)
3. Foundation Methods for foundation models for scientific machine learning — [URL, no paper](https://neurips.cc/virtual/2024/107819)
4. AI-Augmented Climate simulators and emulators — [URL, no paper](https://neurips.cc/virtual/2024/107822)
5. Provable in-context learning of linear systems and linear elliptic PDEs with transformers — [NIPS](https://openreview.net/forum?id=xDstmuxn1D)
6. VSMNO: Solving PDE by Utilizing Spectral Patterns of Different Neural Operators — [NIPS PDF](https://openreview.net/pdf?id=oCT8pYix5e)

### March 2025 Physics Problem Simulations

- The Well: a Large-Scale Collection of Diverse Physics Simulations for ML — [arXiv](https://arxiv.org/pdf/2412.00568) · [Code](https://polymathic-ai.org/the_well/data_format/)
- Polymathic: Advancing Science through Multi-Disciplinary AI — [blog](https://polymathic-ai.org/)
- Long Term Memory: The Foundation of AI Self-Evolution — [arXiv](https://arxiv.org/html/2410.15665v1)
- Distilling Free-Form Natural Laws from Experimental Data (2009) — [Science](https://www.science.org/doi/abs/10.1126/science.1165893) · [comment](https://arxiv.org/pdf/1210.7273) · [medium](https://medium.com/@lotussavy/distilling-free-form-natural-laws-from-experimental-data-f55341ae0fa6)
- Deep learning for universal linear embeddings of nonlinear dynamics — [Nature](https://www.nature.com/articles/s41467-018-07210-0)
- A comparison of data-driven approaches to low-dimensional ocean models (2021) — [arXiv](https://arxiv.org/abs/2108.00818)
- Applications of DL to Ocean Data Inference & Subgrid Parameterization (2018) — [preprint](https://eartharxiv.org/repository/view/1142/)
- On energy-aware hybrid models (2024) — [doi](https://doi.org/10.1029/2024MS004306)

### Spatial-Temporal Graph Modeling

- Graph WaveNet — [arXiv](https://arxiv.org/abs/1906.00121)
- Diffusion Convolutional Recurrent Neural Network (DCRNN) — [ICLR](https://arxiv.org/abs/1707.01926)
- Time-SSM: Simplifying & Unifying State Space Models — [arXiv](https://arxiv.org/pdf/2405.16312)
- State Space Reconstruction for Multivariate Time Series — [arXiv](https://arxiv.org/abs/0809.2220)
- Longitudinal predictive modeling of tau progression — [NeuroImage 2021](https://www.sciencedirect.com/science/article/pii/S1053811921004031?via%3Dihub)

---

## Work arrangements

| **Week** | **Date** | **Theme**                                                                                                                             | **Delivery**            |
| -------: | :------- | :------------------------------------------------------------------------------------------------------------------------------------ | :---------------------- |
|        1 | Sep 4    | Preliminary discussion — [pdf](https://github.com/vadim-vic/Foundation-ts/tree/main/doc/Foundation_models_for_time_series_Week_1.pdf) |                         |
|        2 | Sep 11   | Problem statement — [pdf](https://github.com/vadim-vic/Foundation-ts/tree/main/doc/Foundation_models_for_time_series_Week_2.pdf)      |                         |
|        3 | Sep 18   | Preliminary solution                                                                                                                  | Group talk & discussion |
|        4 | Sep 25   | Minimum deployment                                                                                                                    | Group report            |
|        5 | Oct 4+   | FDA                                                                                                                                   | Personal talks          |
|       13 | Nov 29   | Final discussion                                                                                                                      | Group talks             |

---

## Structure of seminars

The semester lasts **12 weeks**; six alternate weeks are for homework.

- **Odd week:** topic intro + homework theme handout.
- **Every week:** essay discussion; collect improvement list.
- **Odd week:** discuss improved essays; integrate into a joint structure.

### Scoring

Group activity: cross-ranking with **Kemeny median**. Personal talks contribute to score.

---

## Week 3 — Homework 1

1. Form a group.
2. Discuss goals and a solution (\[see the problem statement]).
3. Review solution approaches.
4. Select an LLM-GPT.
5. Run the code; verify it works.

   - Store code in the group repository.
   - Store slides/report as well.

6. Make a 10-minute talk covering:

   - Functionality and architecture of the model.
   - Why you selected this model.
   - Alternative models considered.

### Requirements for the text & discussion

1. Comprehensive explanation of the discussed method/question.
2. Principles only; **no experiments**.
3. \~Two pages.
4. Target reader: 2nd–3rd-year student.
5. One figure is **mandatory**.
6. Brief reference to DL structure is welcome.
7. Talk may be a slide or the text itself.
8. References with DOIs.
9. State how it was generated.
10. Note observed gaps to revisit later.

### Style remarks for the essays

Automatic text generation raises the bar for clarity and authorship. Use generative tools to **train persuasion skills**; write for a **thesis defense committee**.
