# PolarGrad: A Class of Matrix-Gradient Optimizers from a Unifying Preconditioning Perspective

PolarGrad (Polar Gradient methods; Lau et al., 2025) is a class of matrix-gradient optimizers based on the concept of gradient-anisotropy preconditioning in optimization. It has close relation to [Muon](https://github.com/KellerJordan/Muon) (Jordan et al., 2024) and stochastic spectral descent (SSD; Carlson et al., 2015a, 2015b). In addition to being an optimizer for matrix parameters in neural networks, PolarGrad can also be viewed as a preconditioned matrix optimization algorithm for matrix optimization problems such as low-rank matrix factorization/completion. 

The main differences between PolarGrad and Muon/SSD are:
- PolarGrad uses the QDWH (Nakatsukasa et al., 2010) or ZOLO-PD (Nakatsukasa and Freund, 2016) algorithm to compute the polar decomposition of the gradient matrix, while Muon uses the Newton-Schulz iteration to compute the polar decomposition (see the section below for further details). The NS iteration is a matrix iterative polynomial method that computes the polar decomposition of a matrix by iteratively applying a polynomial to the matrix. However, it requires tuning of the coefficients of the polynomial, which can be challenging in practice. PolarGrad also include the nuclear norm (the dual norm of the spectral norm) scaling of the update matrix, which is not present in Muon. The inclusion of such term is necessary for the convergence of optimizers based on polar decomposition for strongly convex and Lipschitz smooth problems with deterministic gradients, as shown in the convergence analysis and the matrix quadratic regression example of PolarGrad (Lau et al., 2025). 
- While SSD also includes the nuclear norm scaling, PolarGrad uses more advanced numerical linear algebra algorithms for polar decomposition than the randomized SVD algorithm used in SSD, namely the QDWH and ZOLO-PD algorithms. 

## Overview
This repository provides implementations of PolarGrad in PyTorch utilizing two more advanced numerical linear algebra algorithms for polar decomposition than the Newton-Schulz (NS) iteration:
1. The [QWDH](https://people.maths.ox.ac.uk/nakatsukasa/publishedpdf/pub3.pdf) algorithm (Nakatsukasa et al., 2010; see [here](https://docs.jax.dev/en/latest/_autosummary/jax.scipy.linalg.polar.html#jax.scipy.linalg.polar) and [here](https://docs.jax.dev/en/latest/_autosummary/jax.lax.linalg.qdwh.html#jax.lax.linalg.qdwh) for implementation in JAX)
2. The [ZOLO-PD](https://people.maths.ox.ac.uk/nakatsukasa/publishedpdf/zoloeigsvd.pdf) algorithm (Nakatsukasa and Freund, 2016; see [here](https://people.maths.ox.ac.uk/nakatsukasa/codes/zolomatlabcodes.zip) for the authors' MATLAB implementation)

These two algorithms, unlike the NS iteration, do not require tuning of the coefficients of the matrix iterative polynomial, and they are more numerically stable (Nakatsukasa and Higham, 2012; Nakatsukasa and Freund, 2016). Hence, they are more suitable for matrix parameters of different sizes and potentially ill-conditioned initializations, making them a better candidate and optimizers based on polar decomposition like PolarGrad and [Muon](https://github.com/KellerJordan/Muon) (Jordan et al., 2024) a drop-in replacement of other adaptive gradient optimizers such as Adam(W). Currently, the QWDH algorithm is particularly more efficient for large matrices, while ZOLO-PD is designed for small to medium-sized matrices.

In particular, with the assist of ChatGPT, we translated these implementations in JAX and MATLAB to PyTorch. Currently, limited by the QR decomposition implementation in PyTorch, mixed precisions such as `bfloat16` are not yet supported. Notice that the current implementation is not optimized for speed and parallelization, although we have also provided a DDP implementation `polar_grad_ddp.py`, following the implementation of [Muon](https://github.com/KellerJordan/Muon). The three main files are:

1. `polar.py`: includes the function `polar` which mimics the JAX [`jax.scipy.linalg.polar` function](https://docs.jax.dev/en/latest/_autosummary/jax.scipy.linalg.polar.html#jax.scipy.linalg.polar), which computes the polar decomposition of a matrix using four possible numerical algorithms. 
    
    i. `method=qdwh`: uses the QDWH algorithm (Nakatsukasa et al., 2010) to compute the polar decomposition of a matrix. This is suitable for large matrices and is more numerically stable than the Newton-Schulz iteration.

    ii. `method=zolo-pd`: uses the ZOLO-PD algorithm (Nakatsukasa and Freund, 2016) to compute the polar decomposition of a matrix. This is suitable for small to medium-sized matrices and is also more numerically stable than the Newton-Schulz iteration.

    iii. `method=ns`: uses the Newton-Schulz (NS) iteration to compute the polar decomposition of a matrix. This might require tuning of the coefficients of the matrix iterative polynomial for different model and layer sizes, which can be challenging in practice. This is the same method used in the Muon optimizer (Jordan et al., 2024), and is adopted from its [GitHub repository](https://github.com/KellerJordan/Muon).

    iv. `method=precond_ns`: uses the preconditioned Newton-Schulz iteration in Lewis et al. (2022) to compute the polar decomposition of a matrix. This is potentially an improved variant of the NS iteration with the need of coefficient tuning, but might still suffer from the stability issue of the NS iteration. We include this method for completeness, but is not heavily tested and not used in the experiments in the paper.


2. `polar_grad.py`: includes the `torch.optim.Optimizer` class `PolarGrad` which implements the PolarGrad optimizer based on the above four numerical polar decomposition algorithms of the gradient matrix. 
    - The argument `polar_first` specifies whether polar-first momentum is used; default is `False` which is similar to the implementation of Muon (Jordan et al., 2024). 
    -   The argument `method` specifies which polar decomposition algorithm to use, and can be one of the following: `qdwh` (cf. `qdwh.py` adopted from its [JAX implementation](https://docs.jax.dev/en/latest/_autosummary/jax.lax.linalg.qdwh.html#jax.lax.linalg.qdwh) `jax.lax.linalg.qdwh`), `zolo-pd` (cf. `zolopd.py` adopted from its [MATLAB implementation](https://people.maths.ox.ac.uk/nakatsukasa/codes/zolomatlabcodes.zip)), `ns` or `precond_ns` (cf. `newton_schulz.py` adopted from Muon's [GitHub repository](https://github.com/KellerJordan/Muon)). The default is `qdwh`, which is suitable for large matrices. 
    - The argument `inner_steps` specifies the number of (inner) steps for either the QDWH algorithm or the NS iteration. The other two algorithms (ZOLO-PD and preconditioned NS) do not require this argument. The default is `2`.
    - The arguments `a`, `b` and `c` specify the coefficients of the matrix iterative polynomial for the NS iteration, which are used only when `method='ns'`. The default values are the same as those in Muon, which are suitable for most cases for hidden layers. However, they can be tuned for different model and layer sizes if necessary.

    The optimizer can be used as follows:
    ```python
    optimizer = PolarGrad(model.parameters(), lr=1e-3, weight_decay=0., momentum=0.9, polar_first=False, method='qdwh', inner_steps=2)
    ```

3. `polar_grad_ddp.py`: includes the `torch.optim.Optimizer` class `PolarGrad` which implements the PolarGrad optimizer based on the above four numerical polar decomposition algorithms of the gradient matrix with `torch.distributed`, following the implementation in Muon's [GitHub repository](https://github.com/KellerJordan/Muon).




## Installation of Required Libraries
Install PyTorch (nightly) accodring to the instructions at https://pytorch.org/get-started/locally/, e.g., for Linux and CUDA 12.6:
```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126
```
Then, install some auxiliary libraries:
```bash
pip install -U numpy matplotlib tqdm fire SciencePlots
```
For correct LaTeX rendering in matplotlib, you might also need to have a LaTeX distribution installed, such as TeX Live (MacTeX) or MikTeX, or disable the LaTeX rendering in matplotlib by setting `rcParams['text.usetex'] = False` and changing some of the plot labels in the code.


## Usage
For small-scale experiments which can be run with CPU, you can run the following commands to test the PolarGrad optimizer on different matrix optimization problems. The `--seed` argument is used to set the random seed for reproducibility.
- Matrix quadratic regression (a strongly convex problem with deterministic gradient):
    ```
    # PolarGrad
    python mat_quad_reg.py --steps=4000 --seed=42

    # PolarGradM
    python mat_quad_reg_mom.py --steps=4000 --seed=42
    ```

- Matrix logistic regression (a strongly convex problem with stochastic gradient):
    ```
    # PolarSGD
    python mat_log_reg.py --seed=42

    # PolarSGDM
    python mat_log_reg_mom.py --seed=42
    ```

- Low-rank matrix completion (a non-convex problem with deterministic gradient):
    ```
    # PolarGrad
    python low_rank_mat_comp.py --seed=42

    # PolarGradM
    python low_rank_mat_comp_mom.py --seed=42
    ```

We will update the repository with examples and experiments for language model pre-training soon. 


## Citation
If you find this repository useful for your research, please consider citing the following paper using the BibTeX entry below:
```
@article{lau2025polargrad,
	title={\textsc{PolarGrad}: A Class of Matrix-Gradient Optimizers from a Unifying Preconditioning Perspective},
	author={Lau, Tim Tsz-Kit and Qi Long and Weijie Su},
	year={2025},
	journal={arXiv preprint arXiv:2505.21799}
}
```

## References
-  Lau, Tim Tsz-Kit, Qi Long, and Weijie Su. [PolarGrad: A class of matrix-gradient optimizers from a unifying preconditioning perspective](https://arxiv.org/abs/2505.21799). *arXiv preprint arXiv:2505.21799*, 2025. 

- Jordan, Keller, Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista, Laker Newhouse, and Jeremy Bernstein. [Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/). 2024. 

- Carlson, David, Volkan Cevher, and Lawrence Carin. [Stochastic spectral descent for restricted Boltzmann machines](https://proceedings.mlr.press/v38/carlson15.html). In
*Proceedings of the International Conference on Artificial Intelligence and Statistics (AISTATS)*, 2015a. 

- Carlson, David E., Edo Collins, Ya-Ping Hsieh, Lawrence Carin, and Volkan Cevher. [Preconditioned spectral descent for deep learning](https://papers.nips.cc/paper_files/paper/2015/hash/f50a6c02a3fc5a3a5d4d9391f05f3efc-Abstract.html). In *Advances in Neural Information Processing Systems (NeurIPS)*, 2015b.

- Nakatsukasa, Yuji, Zhaojun Bai, and François Gygi. [Optimizing Halley's iteration for computing the matrix polar decomposition](https://doi.org/10.1137/090774999). *SIAM Journal on Matrix Analysis and Applications*, 31(5):2700-2720, 2010.

- Nakatsukasa, Yuji and Roland W. Freund. [Computing fundamental matrix decompositions accurately via the matrix sign function in two iterations: The power of Zolotarev's functions](https://doi.org/10.1137/140990334). *SIAM Review*, 58(3):461-493, 2016. 

- Nakatsukasa, Yuji, and Nicholas J. Higham. [Backward stability of iterations for computing the polar decomposition](https://doi.org/10.1137/110857544). *SIAM Journal on Matrix Analysis and Applications*, 33(2):460-479, 2012.

- Lewis, Adam G. M., Jackson Beall, Martin Ganahl, Markus Hauru, Shrestha Basu Mallick, and Guifre Vidal. [Large-scale distributed linear algebra with tensor processing units](). *Proceedings of the National Academy of Sciences*, 119(33):e2122762119, 2022. 