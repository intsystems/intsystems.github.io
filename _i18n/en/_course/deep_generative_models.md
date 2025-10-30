### About

The course is devoted to modern generative models (mostly in the application to computer vision). Special attention is paid to the properties of various classes of generative models, their interrelationships, theoretical prerequisites and methods of quality assessment. The aim of the course is to introduce the student to widely used advanced methods of deep learning.

### Syllabus

1. Generative models overview and motivation. Problem statement. Divergence minimization framework. Autoregressive models (PixelCNN).
2. Normalizing Flow (NF) intuition and definition. Linear NF. Gaussian autoregressive NF. Coupling layer (RealNVP).
3. Forward and reverse KL divergence for NF. Latent variable models (LVM). Variational lower bound (ELBO). EM-algorithm.
4. Amortized inference, ELBO gradients, reparametrization trick. Variational Autoencoder (VAE). NF as VAE model. Discrete VAE latent representations.
5. Vector quantization, straight-through gradient estimation (VQ-VAE). Gumbel-softmax trick (DALL-E). ELBO surgery and optimal VAE prior. Learnable VAE prior.
6. Likelihood-free learning. GAN optimality theorem. Wasserstein distance.
7. Wasserstein GAN (WGAN). f-divergence minimization. GAN evaluation (FID, Precision-Recall, truncation trick).
8. Langevin dynamic. Score matching (Denoising score matching, Noise Conditioned Score Network (NCSN)). Forward gaussian diffusion process.
9. Denoising score matching for diffusion. Reverse Gaussian diffusion process. Gaussian diffusion model as VAE. ELBO for DDPM.
10. Denoising diffusion probabilistic model (DDPM): reparametrization and overview. Denoising diffusion as score-based generative model. Model guidance: classifier guidance, classfier-free guidance.
11. Continuous-in-time NF and neural ODE. Continuity equation for NF log-likelihood. FFJORD and Hutchinson's trace estimator. Adjoint method for continuous-in-time NF.
12. SDE basics. Kolmogorov-Fokker-Planck equation. Probability flow ODE. Reverse SDE. Variance Preserving and Variance Exploding SDEs.
13. Score-based generative models through SDE. Flow matching. Conditional flow matching. Conical gaussian paths.
14. Linear interpolation. Link with diffusion and score matching. Latent space models.

### Labworks

6 homeworks: theory and practice.

### Grading

Each homework gives 15 points + an exam for 30 points. Final score: `min(floor(#points/10), 10)`.

### Prerequisites

Probability theory, statistics, machine learning, basics of deep learning, python, pytorch, intro to bayesian inference.
