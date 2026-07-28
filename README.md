# Learning Parametrized Hamiltonians

Small, reproducible C++ benchmarks for learning a time-dependent Hamiltonian
from noisy queries to its unitary dynamics.

## Benchmark algorithms

The test system is

$$
H(t)=a(x)\,Z\otimes I+b(x)\,I\otimes X+c(x)\,Y\otimes Z,\qquad
x=2t/T-1,\quad T=2,
$$

with

$$
\begin{aligned}
a(x)&=0.60+0.25T_1(x)-0.10T_3(x),\\
b(x)&=-0.45+0.20T_2(x)+0.12T_5(x),\\
c(x)&=0.50+0.18T_1(x)-0.14T_4(x).
\end{aligned}
$$

The reference unitary solves
$\dot U(t)=-iH(t)U(t)$, $U(0)=I$, using high-accuracy RK4 integration.
At the $M$ roots of the first-kind Chebyshev polynomial, the learner receives

$$
\widetilde U(t_j)=U(t_j)+N_j,\qquad
(N_j)_{kl}\sim\mathcal{CN}(0,\sigma^2).
$$

The default sweep uses
$\sigma\in\{0,0.001,0.0025,0.005,0.01,0.02\}$, corresponding to RMS relative
Frobenius noise $2\sigma\in\{0,0.2,0.5,1,2,4\}\%$ for a two-qubit unitary.
The seed is fixed to `20260727`, and the program reports
$M\in\{3,5,8,12,16\}$. For fixed $M$, the same Gaussian draw is scaled across
all noise rates; both learners always receive identical noisy matrices.

Both algorithms:

1. interpolate every matrix entry of $\widetilde U(t)$ in a degree-$M-1$
   Chebyshev series;
2. differentiate the series analytically;
3. form $i\widehat U'(t)\widehat U(t)^\dagger$, then project it onto the
   Hermitian, traceless matrices.

The **unconstrained** baseline applies these steps directly to the noisy
samples. The **polar-projected** learner first replaces every noisy sample
$A=\widetilde U(t_j)$ by its unitary polar factor

$$
\mathcal P_{U(4)}(A)=A(A^\dagger A)^{-1/2}.
$$

For nonsingular $A$, this is the unique minimizer of
$\min_{Q\in U(4)}\|A-Q\|_F$. The implementation computes the polar factor by
the quadratically convergent Newton iteration
$X_{k+1}=(X_k+X_k^{-\dagger})/2$. Only the queried matrices are projected;
the Chebyshev interpolant is not constrained to be unitary between nodes.

Geometrically, the leading-order perturbation of a unitary splits into
tangent and normal components. The polar projection retains the tangent
component and removes the normal component. For isotropic complex Gaussian
noise, these two real subspaces both have dimension 16 in the ambient
32-dimensional space of complex $4\times4$ matrices, so the sample-level
mean-squared error is halved to leading order. This does not imply a factor
of two improvement after interpolation and differentiation: the final
Hermitian/traceless generator projection already rejects part of the
irrelevant perturbation.

The reported error is the uniform-time normalized $L_2$-Frobenius error

$$
E_H=
\left[
\frac{\int_0^T\|\widehat H(t)-H(t)\|_F^2\,dt}
{\int_0^T\|H(t)\|_F^2\,dt}
\right]^{1/2},
$$

evaluated by 256-point Gauss-Legendre quadrature. Both Hamiltonians are
traceless by construction.

The primary runtime is intrinsic learner construction: sample projection
(polar-projected learner only), Chebyshev fitting, and analytic
differentiation. The CSV separately records the common cost of reconstructing
$\widehat H(t)$ at the 256 quadrature nodes used by the evaluation protocol;
this grid-dependent cost is not included in the primary runtime. Reference
dynamics, synthetic-noise generation, and ground-truth error accumulation are
excluded. Sub-millisecond timings are averaged over repeated identical
operations, taking the best of five batches to suppress scheduler
interruptions.

## Fixed-seed results

Accuracy for seed `20260727`. Each entry is
unconstrained / polar-projected:

| $\sigma$ | RMS noise | $M=3$ | $M=5$ | $M=8$ | $M=12$ | $M=16$ |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0% | .30701 / .30701 | .15090 / .15090 | .01292 / .01292 | .00016 / .00016 | .00001 / .00001 |
| .001 | .2% | .30692 / .30699 | .15000 / .15013 | .01707 / .01684 | .01892 / .01864 | .02342 / .02285 |
| .0025 | .5% | .30684 / .30700 | .14925 / .14952 | .03032 / .02986 | .04728 / .04661 | .05853 / .05713 |
| .005 | 1% | .30682 / .30715 | .14956 / .14998 | .05598 / .05535 | .09448 / .09325 | .11693 / .11424 |
| .01 | 2% | .30725 / .30790 | .15585 / .15632 | .10907 / .10855 | .18859 / .18664 | .23341 / .22839 |
| .02 | 4% | .30990 / .31125 | .18689 / .18724 | .21533 / .21664 | .37580 / .37378 | .46512 / .45641 |

Intrinsic learner-construction runtime on the benchmark machine:

| $M$ | Unconstrained [$\mu$s] | Polar-projected [$\mu$s] |
|---:|---:|---:|
| 3 | 0.509 | 7.400 |
| 5 | 1.262 | 12.938 |
| 8 | 3.018 | 21.594 |
| 12 | 6.477 | 34.068 |
| 16 | 11.287 | 48.761 |

At $M=3,5$, interpolation bias dominates and projection is slightly
counterproductive for this realization. It consistently helps at $M=12,16$.
The intermediate $M=8$ crosses over: projection helps through 2% RMS input
noise but is slightly worse at 4%. Thus sample-wise unitarity is a useful but
modest denoising constraint, not a uniformly dominating estimator.

## Build and run

The implementation is dependency-free apart from a C++20 compiler and CMake.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/lph_benchmark
```

The default run writes `results/chebyshev_noise_sweep.csv`. Passing
`--sigma X` restricts the run to one noise rate. Use `--help` to see the
remaining reproducibility and accuracy options.
