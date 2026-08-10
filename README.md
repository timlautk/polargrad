# PolarGrad: A Class of Matrix-Gradient Optimizers from a Unifying Preconditioning Perspective

[PolarGrad](https://arxiv.org/abs/2505.21799) (Polar Gradient methods; Lau et al., 2025) is a class of matrix-gradient optimizers based on the concept of gradient-anisotropy preconditioning in optimization. It has close relation to [Muon](https://github.com/KellerJordan/Muon) (Jordan et al., 2024) and stochastic spectral descent (SSD; Carlson et al., 2015a, 2015b). In addition to being an optimizer for matrix parameters in neural networks, PolarGrad can also be used as a preconditioned matrix optimization algorithm for matrix optimization problems such as matrix regression and low-rank matrix factorization/completion. 

The main differences between PolarGrad and Muon/SSD are:
- PolarGrad uses the QDWH (Nakatsukasa et al., 2010) or ZOLO-PD (Nakatsukasa and Freund, 2016) algorithm to compute the polar decomposition of the gradient matrix, while Muon uses the Newton-Schulz iteration to compute the polar decomposition (see the section below for further details). The NS iteration is a matrix iterative polynomial method that computes the polar decomposition of a matrix by iteratively applying a polynomial to the matrix. However, it requires tuning of the coefficients of the polynomial, which can be challenging in practice. PolarGrad also include the nuclear norm (the dual norm of the spectral norm) scaling of the update matrix, which is not present in Muon. The inclusion of such term is necessary for the convergence of optimizers based on polar decomposition for strongly convex and Lipschitz smooth problems with deterministic gradients, as shown in the convergence analysis and the matrix quadratic regression example of PolarGrad (Lau et al., 2025). 
- Following the concurrent work of [Amsel et al. (2025)](https://arxiv.org/abs/2505.16932), PolarGrad also includes the Polar Express method to compute the polar decomposition of a matrix, which uses polynomial approximations of the sign function to compute the polar decomposition, rather than rational approximations of the sign function used in the QDWH and ZOLO-PD algorithms, hence avoiding the use of QR decompositions and involving only matrix-matrix products (in half-precision arithmetic). Its implementation is directly taken from the paper. Yet, it has not been heavily tested in experiments yet. 
- While SSD also includes the nuclear norm scaling, PolarGrad uses more advanced numerical linear algebra algorithms for polar decomposition than the randomized SVD algorithm used in SSD, namely the QDWH and ZOLO-PD algorithms. 

## Overview
This repository provides implementations of PolarGrad in PyTorch utilizing two more advanced numerical linear algebra algorithms for polar decomposition than the Newton-Schulz (NS) iteration:
1. The [QWDH](https://people.maths.ox.ac.uk/nakatsukasa/publishedpdf/pub3.pdf) algorithm (Nakatsukasa et al., 2010; see [here](https://docs.jax.dev/en/latest/_autosummary/jax.scipy.linalg.polar.html#jax.scipy.linalg.polar) and [here](https://docs.jax.dev/en/latest/_autosummary/jax.lax.linalg.qdwh.html#jax.lax.linalg.qdwh) for implementation in JAX)
2. The [ZOLO-PD](https://people.maths.ox.ac.uk/nakatsukasa/publishedpdf/zoloeigsvd.pdf) algorithm (Nakatsukasa and Freund, 2016; see [here](https://people.maths.ox.ac.uk/nakatsukasa/codes/zolomatlabcodes.zip) for the authors' MATLAB implementation)

These two algorithms, unlike the NS iteration, do not require tuning of the coefficients of the matrix iterative polynomial, and they are more numerically stable (Nakatsukasa and Higham, 2012; Nakatsukasa and Freund, 2016). Hence, they are more suitable for matrix parameters of different sizes and potentially ill-conditioned initializations, making them a better candidate and optimizers based on polar decomposition like PolarGrad and [Muon](https://github.com/KellerJordan/Muon) (Jordan et al., 2024) a drop-in replacement of other adaptive gradient optimizers such as Adam(W). Currently, the QWDH algorithm is particularly more efficient for large matrices, while ZOLO-PD is designed for small to medium-sized matrices. Note that both of these algorithms involve QR decompositions, which might not be efficient for GPUs and half-precision arithmetic. To addresss such issue, we also include the Polar Express method in [Amsel et al. (2025)](https://arxiv.org/abs/2505.16932) to compute the polar decomposition of a matrix, which uses polynomial approximations of the sign function to compute the polar decomposition, rather than rational approximations of the sign function used in the QDWH and ZOLO-PD algorithms, hence avoiding the use of QR decompositions and involving only matrix-matrix products (in half-precision arithmetic). 

In particular, with the assist of ChatGPT, we translated these implementations in JAX and MATLAB to PyTorch. Currently, limited by the QR decomposition implementation in PyTorch, mixed precisions such as `bfloat16` are not yet supported. Notice that the current implementation is not optimized for speed and parallelization, although we have also provided a DDP implementation `polar_grad_ddp.py`, following the implementation of [Muon](https://github.com/KellerJordan/Muon). The three main files are:

1. `polar.py`: includes the function `polar` which mimics the JAX [`jax.scipy.linalg.polar` function](https://docs.jax.dev/en/latest/_autosummary/jax.scipy.linalg.polar.html#jax.scipy.linalg.polar), which computes the polar decomposition of a matrix using four possible numerical algorithms. 
    
    i. `method=qdwh`: uses the QDWH algorithm (Nakatsukasa et al., 2010) to compute the polar decomposition of a matrix. This is suitable for large matrices and is more numerically stable than the Newton-Schulz iteration.

    ii. `method=zolo-pd`: uses the ZOLO-PD algorithm (Nakatsukasa and Freund, 2016) to compute the polar decomposition of a matrix. This is suitable for small to medium-sized matrices and is also more numerically stable than the Newton-Schulz iteration.

    iii. `method=ns`: uses the Newton-Schulz (NS) iteration to compute the polar decomposition of a matrix. This might require tuning of the coefficients of the matrix iterative polynomial for different model and layer sizes, which can be challenging in practice. This is the same method used in the Muon optimizer (Jordan et al., 2024), and is adopted from its [GitHub repository](https://github.com/KellerJordan/Muon).

    iv. `method=precond_ns`: uses the preconditioned Newton-Schulz iteration in [Lewis et al. (2022)](https://doi.org/10.1073/pnas.2122762119) to compute the polar decomposition of a matrix. This is potentially an improved variant of the NS iteration with the need of coefficient tuning, but might still suffer from the stability issue of the NS iteration. We include this method for completeness, but is not heavily tested and not used in the experiments in the paper.

    v. `method=polar_express`: uses the Polar Express method in [Amsel et al. (2025)](https://arxiv.org/abs/2505.16932) to compute the polar decomposition of a matrix. 

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
    python mat_log_reg.py --steps=1500 --seed=42

    # PolarSGDM
    python mat_log_reg_mom.py --steps=1500 --seed=42
    ```

- Low-rank matrix completion (a non-convex problem with deterministic gradient):
    ```
    # PolarGrad
    python low_rank_mat_comp.py --steps=1000 --seed=42

    # PolarGradM
    python low_rank_mat_comp_mom.py --steps=200 --seed=42
    ```

We will update the repository with examples and experiments for language model pre-training soon. 

## Timing, memory, and GPU benchmarks for Sections 6.1--6.3

The three main experiment scripts have an opt-in benchmark mode. The benchmark
uses a fresh copy of each workload and omits the condition-number and
nuclear-norm diagnostics from the timed region. This is important because those
diagnostics can cost substantially more than the optimizer step. A throwaway
warmup run absorbs lazy CUDA initialization and `torch.compile` overhead, while
the measured run starts again from the same seed and model initialization.
Commit the benchmark code before collecting final numbers. Publication runs
refuse a dirty worktree by default, record a SHA-256 fingerprint of every Python
source file, and maintain an output-directory manifest that prevents results
from different commits, source trees, PyTorch builds, GPUs, or matmul-precision
settings from being mixed accidentally. `--allow_dirty_git=True` and
`--allow_mixed_runs=True` are development-only escape hatches.

First run the numerical regression checks:

```bash
python validate_polar_oracles.py --device=cpu
```

To run only the timing and memory measurements on a CUDA GPU:

```bash
python mat_quad_reg.py --device=cuda --seed=42 --benchmark_only=True --benchmark_repeats=3 --benchmark_trace_every=10 --results_dir=results_corrected
python mat_quad_reg.py --device=cuda --seed=142 --benchmark_only=True --benchmark_repeats=3 --benchmark_trace_every=10 --results_dir=results_corrected
python mat_quad_reg.py --device=cuda --seed=242 --benchmark_only=True --benchmark_repeats=3 --benchmark_trace_every=10 --results_dir=results_corrected
python mat_log_reg.py --device=cuda --seed=42 --benchmark_only=True --benchmark_repeats=3 --benchmark_trace_every=10 --results_dir=results_corrected
python mat_log_reg.py --device=cuda --seed=142 --benchmark_only=True --benchmark_repeats=3 --benchmark_trace_every=10 --results_dir=results_corrected
python mat_log_reg.py --device=cuda --seed=242 --benchmark_only=True --benchmark_repeats=3 --benchmark_trace_every=10 --results_dir=results_corrected
python low_rank_mat_comp.py --device=cuda --seed=42 --benchmark_only=True --benchmark_repeats=3 --benchmark_trace_every=10 --results_dir=results_corrected
python low_rank_mat_comp.py --device=cuda --seed=142 --benchmark_only=True --benchmark_repeats=3 --benchmark_trace_every=10 --results_dir=results_corrected
python low_rank_mat_comp.py --device=cuda --seed=242 --benchmark_only=True --benchmark_repeats=3 --benchmark_trace_every=10 --results_dir=results_corrected
```

Use `--benchmark=True` instead of `--benchmark_only=True` to generate the
original iteration-based figures first and then run the independent benchmark.
By default, the benchmark measures the same number of steps as the experiment.
For a short validation run, pass (for example) `--benchmark_steps=100`. The
options `--benchmark_warmup`, `--benchmark_repeats`,
`--benchmark_trace_every`, and `--results_dir` control the warmup length,
number of fresh timing repeats, independent convergence-trace interval, and
output directory.

For targeted development or hardware profiling, `--benchmark_filter` accepts
a comma-separated list of stable method identifiers. For example,
`--benchmark_filter=polargrad_qdwh_lr_decay` runs only decayed QDWH PolarGrad.
An invalid identifier prints all available choices. Passing
`--benchmark_nvtx=True` places only each measured repetition inside an NVTX
range of the form
`training/<experiment>/<method_identifier>/repeat=<index>`; warmup, final
metrics, and convergence diagnostics remain outside the range. Use
`--benchmark_trace_every=0` while profiling so no separate trace is collected.

Each script writes JSON, per-repeat CSV, summary CSV, and convergence-trace CSV
files. Final objectives are evaluated after the last update and outside the
timed region. Quadratic regression also records the exact objective gap;
logistic regression records a deterministic full-data objective rather than the
last random mini-batch. The separate trace makes fixed-objective threshold
comparisons possible without inserting loss synchronizations into the timed
repetitions.

Matrix completion additionally records observed-entry loss, unobserved-entry
loss, full-matrix mean squared error, and relative Frobenius reconstruction
error. The observed-entry loss remains available as `final_loss` for backward
compatibility. The unobserved and full-matrix metrics measure recovery rather
than merely fitting the entries used for optimization.

After all seeds finish, validate and aggregate the runs with:

```bash
python summarize_benchmark_runs.py --results-dir=results_corrected
```

Optional time-to-threshold summaries can be generated from the independent
traces, for example with
`--threshold=mat_quad_reg:objective_gap:1e-2`. The reported wall time is the
threshold-crossing step multiplied by the median step time from the independent
timing repetitions and is therefore labeled as an estimate.

The timing records include total wall time, time per step, steps per second,
median sampled forward/backward and optimizer times, optimizer-state size,
baseline CUDA allocation, and peak allocated/reserved CUDA memory. The
estimated temporary workspace is the incremental peak minus the final
optimizer-state tensor size and remains allocator-dependent. The field
`cuda_stream_span_fraction_pct` is only the duration of the measured CUDA
stream span divided by wall time. It is not GPU utilization, SM occupancy, or
achieved FLOP efficiency and must not be reported as "GPU efficiency."

The polar-oracle microbenchmark compares fixed inner-iteration budgets while
also reporting orthogonality, reconstruction, and optional exact-direction
errors:

```bash
python benchmark_polar_oracles.py \
  --device=cuda \
  --shapes=500x100,1000x100,500x5,250x5,1024x1024,4096x1024 \
  --methods=qdwh,zolo-pd,ns,polar_express \
  --spectra=gaussian,ill_conditioned,rank_deficient \
  --inner-steps=2,5 \
  --condition-number=1e6 \
  --compute-reference \
  --matmul-precision=highest \
  --repeats=3 \
  --output-dir=results_corrected/oracles
```

The oracle residuals are evaluated in float64 and include orthogonality,
Hermitian symmetry, positive-semidefiniteness, reconstruction with a
symmetrized Hermitian factor, the polar-objective gap, and (for full-rank
matrices) direction error against a float64 SVD. Direction error is intentionally
omitted for rank-deficient inputs because the polar factor is nonunique on the
null space.

Add `--profile` to save PyTorch Chrome traces. For publication-quality hardware
utilization, run one representative configuration at a time under NVIDIA
Nsight Compute. `--nvtx` annotates only the measured call block. For example:

```bash
ncu --target-processes all --nvtx --nvtx-include "polar/polar_express/steps=5/4096x1024/gaussian/" --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed --export results_corrected/ncu_polar_express_4096x1024 python benchmark_polar_oracles.py --device=cuda --shapes=4096x1024 --methods=polar_express --spectra=gaussian --inner-steps=5 --calls=20 --warmup-calls=5 --repeats=1 --compute-reference --nvtx --output-dir=results_corrected/ncu_polar_express
```

Report duration-weighted SM-throughput and DRAM-throughput percentages and
state their Nsight metric names explicitly. They should not be inferred from
CUDA event timing.

End-to-end training ranges can be profiled in the same way. A representative
quadratic-regression command is:

```bash
ncu --target-processes all --nvtx --nvtx-include "training/mat_quad_reg/polargrad_qdwh_lr_decay/repeat=0/" --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed --export results_additional/ncu_mat_quad_polargrad_qdwh_decay --force-overwrite python mat_quad_reg.py --device=cuda --seed=42 --benchmark_only=True --benchmark_steps=50 --benchmark_warmup=10 --benchmark_repeats=1 --benchmark_trace_every=0 --benchmark_filter=polargrad_qdwh_lr_decay --benchmark_nvtx=True --results_dir=results_additional/mat_quad_polargrad_qdwh_decay
```

Use a fresh results directory after modifying the benchmark source. The schema
and source-hash checks intentionally reject attempts to append new records to
an older results directory.

The complete set of representative end-to-end and oracle Nsight Compute runs
is available as a script:

```bash
./run_additional_benchmarks.sh training
./run_additional_benchmarks.sh oracles
# Or run both groups:
./run_additional_benchmarks.sh all
```

By default, `.ncu-rep` files are saved under `results_additional/ncu`. Set
`POLARGRAD_PROFILE_ROOT` to choose another root directory.

If only the new matrix-recovery metrics are needed, rerun the three completion
seeds in their own clean directory and aggregate only that experiment:

```bash
python low_rank_mat_comp.py --device=cuda --seed=42 --benchmark_only=True --benchmark_repeats=3 --benchmark_trace_every=10 --results_dir=results_recovery_v2_1
python low_rank_mat_comp.py --device=cuda --seed=142 --benchmark_only=True --benchmark_repeats=3 --benchmark_trace_every=10 --results_dir=results_recovery_v2_1
python low_rank_mat_comp.py --device=cuda --seed=242 --benchmark_only=True --benchmark_repeats=3 --benchmark_trace_every=10 --results_dir=results_recovery_v2_1
python summarize_benchmark_runs.py --results-dir=results_recovery_v2_1 --expected-experiments=low_rank_mat_comp
```

Reproducibility corrections made with this benchmark update:

- The experiment scripts now pass the reported inner-iteration budgets
  explicitly: two QDWH steps and five Newton--Schulz steps. Previously,
  `Muon_polar` silently used its default of five steps for QDWH.
- Matrix logistic regression now generates `C` by thresholding standard
  Gaussian samples at `0.5` and encoding the two classes as `-1` and `+1`, as
  required by the stated loss `softplus(-C * logits)`. The former `0/1`
  encoding made every zero-labeled entry contribute a constant with zero
  gradient.
- Low-rank matrix completion now applies the observation mask to the residual
  in both the joint optimizer and AltGD objectives. Previously, the mask was
  generated but used only in the denominator.
- QDWH and ZOLO-PD now compute Hermitian right solves with
  `torch.cholesky_solve`. The previous QDWH code treated an adjoint Cholesky
  factor as lower triangular, and the previous ZOLO-PD code obtained a lower
  Cholesky factor but solved as though it were upper triangular.
- Polar Express and Newton--Schulz now validate one-dimensional inputs before
  using `.mT`, and their optional Hermitian-factor calculation works for both
  tall and wide matrices.


## Citation
If you find this repository useful for your research, please consider citing our paper using the BibTeX entry below:
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

- Lewis, Adam G. M., Jackson Beall, Martin Ganahl, Markus Hauru, Shrestha Basu Mallick, and Guifre Vidal. [Large-scale distributed linear algebra with tensor processing units](https://doi.org/10.1073/pnas.2122762119). *Proceedings of the National Academy of Sciences*, 119(33):e2122762119, 2022. 

- Amsel, Noah, David Persson, Christopher Musco, and Robert M. Gower. [The Polar Express: Optimal matrix sign methods and their application to the Muon algorithm](https://arxiv.org/abs/2505.16932). *arXiv preprint arXiv:2505.16932*, 2025.
