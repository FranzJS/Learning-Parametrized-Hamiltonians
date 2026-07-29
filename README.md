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
$\sigma\in\{0,0.001,0.0025,0.005,0.01,0.02,0.05,0.1,0.2,0.35,0.5,1\}$,
corresponding to RMS relative Frobenius noise
$2\sigma\in\{0,0.2,0.5,1,2,4,10,20,40,70,100,200\}\%$ for a two-qubit
unitary. Rates at and above 40% are deliberately extreme stress tests rather
than representative tomography noise levels.
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
| .05 | 10% | .33061 / .33574 | .34399 / .34943 | .52793 / .54592 | .93092 / .93718 | 1.15281 / 1.13751 |
| .1 | 20% | .39543 / .41586 | .64357 / .67017 | 1.03197 / 1.10543 | 1.84531 / 1.87598 | 2.29132 / 2.25760 |
| .2 | 40% | .57653 / .64339 | 1.24377 / 1.26714 | 2.02738 / 2.23416 | 3.71704 / 3.68461 | 4.66903 / 4.38978 |
| .35 | 70% | .90769 / .98882 | 2.20578 / 1.81763 | 3.70946 / 3.68661 | 6.96172 / 5.95628 | 8.97678 / 7.07563 |
| .5 | 100% | 1.33461 / 1.22433 | 3.40045 / 2.10290 | 5.91736 / 4.27403 | 11.15048 / 7.02437 | 14.74838 / 8.51520 |
| 1 | 200% | 3.95435 / 1.42825 | 10.40155 / 2.47527 | 19.06890 / 4.43601 | 34.75665 / 7.99427 | 47.91194 / 10.97079 |

Intrinsic learner-construction runtime on the benchmark machine:

| $M$ | Unconstrained [$\mu$s] | Polar-projected [$\mu$s] |
|---:|---:|---:|
| 3 | 0.509 | 7.400 |
| 5 | 1.262 | 12.938 |
| 8 | 3.018 | 21.594 |
| 12 | 6.477 | 34.068 |
| 16 | 11.287 | 48.761 |

In the low-noise regime, projection is a modest and nonuniform improvement:
interpolation bias dominates for small $M$, while differentiation amplifies
noise for large $M$. At 10--40% RMS noise, projection can still be worse,
depending on $M$. It becomes decisively useful only in the extreme regime
where the additive noise is comparable to or larger than the unitary signal.
Even there the absolute Hamiltonian errors remain large; projection prevents
the estimator from exploding but does not recover an accurate Hamiltonian.

## Four-qubit benchmark

The connected four-qubit extension is

$$
\begin{aligned}
H_4(t)={}&a(x)(Z_1+Z_3)+b(x)(X_2+X_4)\\
&+c(x)(Y_1Z_2+Y_3Z_4)\\
&+\frac{d(x)}{\sqrt3}
\left(X_2X_3+Y_2Y_3+Z_2Z_3\right),
\end{aligned}
$$

where $a,b,c$ are unchanged and

$$
d(x)=0.35-0.15T_2(x)+0.10T_4(x).
$$

The bridge between qubits 2 and 3 makes the dynamics nonfactorizing while
preserving nearest-neighbour 2-locality. The four-qubit sweep uses relative
RMS Frobenius noise

$$
r=\sqrt{\frac{\mathbb E\lVert N\rVert_F^2}{\lVert U\rVert_F^2}}
\in\{0.5,1,2,3\}\%,
\qquad \sigma=\frac r{\sqrt{16}}=\frac r4.
$$

The four-qubit executable adds a third, **unitary-curve** learner. Starting
from the sample-wise polar factors
$Q_j=\mathcal P_{U(16)}(\widetilde U(t_j))$, it constructs their ordinary
matrix-valued Chebyshev interpolant $P_M(t)$ and defines

$$
\widehat U_{\rm curve}(t)=\mathcal P_{U(16)}(P_M(t)).
$$

Thus the fitted curve is unitary at every point where $P_M(t)$ is
nonsingular. It still interpolates the $Q_j$ exactly, and at every
intermediate time it is the unique Frobenius-nearest unitary to the
unconstrained continuation $P_M(t)$. This is a pointwise orthogonal-projection
best fit, not a global nonlinear least-squares optimization over all unitary
paths.

The derivative is analytic. The implementation differentiates the polar
Newton iteration by evolving $X_0=P_M$, $Y_0=\dot P_M$ according to

$$
\begin{aligned}
X_{k+1}&=\frac12\left(X_k+X_k^{-\dagger}\right),\\
Y_{k+1}&=\frac12\left(
Y_k-X_k^{-\dagger}Y_k^\dagger X_k^{-\dagger}
\right).
\end{aligned}
$$

At convergence, $X_k=\widehat U_{\rm curve}$ and
$Y_k=\dot{\widehat U}_{\rm curve}$. The estimate is therefore

$$
\widehat H_{\rm curve}(t)=
\Pi_{\rm Herm,0}\!\left(
iY_kX_k^\dagger
\right).
$$

No finite-difference step or additional fitted hyperparameter is used.

Accuracy for seed `20260727`; each entry is
unconstrained / sample-polar / unitary-curve:

| Relative noise | $M=3$ | $M=5$ | $M=8$ | $M=12$ | $M=16$ |
|---:|---:|---:|---:|---:|---:|
| 0.5% | .48909 / .48932 / .50054 | .21093 / .21068 / .20925 | .03141 / .03122 / .03088 | .03183 / .03123 / .03138 | .04315 / .04242 / .04255 |
| 1% | .48891 / .48935 / .50091 | .21184 / .21125 / .20969 | .04503 / .04420 / .04421 | .06360 / .06237 / .06268 | .08630 / .08483 / .08509 |
| 2% | .48874 / .48953 / .50183 | .21520 / .21366 / .21196 | .07859 / .07638 / .07702 | .12719 / .12463 / .12527 | .17261 / .16959 / .17012 |
| 3% | .48882 / .48988 / .50300 | .22052 / .21773 / .21600 | .11436 / .11080 / .11202 | .19082 / .18683 / .18781 | .25897 / .25429 / .25508 |

Median learner-construction runtime across the four noise rates:

| $M$ | Unconstrained [$\mu$s] | Sample-polar [$\mu$s] | Unitary-curve [$\mu$s] |
|---:|---:|---:|---:|
| 3 | 6.451 | 183.367 | 182.479 |
| 5 | 15.473 | 307.959 | 309.832 |
| 8 | 34.983 | 505.621 | 506.193 |
| 12 | 73.846 | 788.127 | 791.876 |
| 16 | 126.667 | 1142.956 | 1141.486 |

Median runtime to reconstruct $\widehat H$ at all 256 error-quadrature nodes:

| $M$ | Unconstrained [ms] | Sample-polar [ms] | Unitary-curve [ms] |
|---:|---:|---:|---:|
| 3 | 2.461 | 2.428 | 47.215 |
| 5 | 2.931 | 2.781 | 41.503 |
| 8 | 3.478 | 3.629 | 34.247 |
| 12 | 4.525 | 4.185 | 35.413 |
| 16 | 5.139 | 5.199 | 36.334 |

Synthetic data generation includes RK4 evolution to all query times and
addition of the Gaussian noise. It is shared by all three algorithms and does
not depend materially on the noise rate:

| $M$ | Data generation [ms] |
|---:|---:|
| 3 | 158.88 |
| 5 | 166.53 |
| 8 | 166.98 |
| 12 | 172.29 |
| 16 | 163.94 |

The CSV records construction, 256-node evaluation, and their sum separately.
The 256-node cost is an evaluation-protocol choice rather than an intrinsic
fit cost, but it is important for the unitary-curve learner because polar
projection and its derivative are evaluated on demand. The data-generation
implementation exploits the sparse Pauli form of $H_4(t)$; all absolute
runtimes are machine-dependent. Halving the RK4 step from $2\cdot10^{-4}$ to
$10^{-4}$ changes every reported error by at most $1.6\cdot10^{-14}$.
All fitted curves passed unitarity and tangent-space checks on a uniform
1,025-point validation grid.

## Build and run

The implementation is dependency-free apart from a C++20 compiler and CMake.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/lph_benchmark
./build/lph_four_qubit_benchmark
```

The default run writes `results/chebyshev_noise_sweep.csv`. Passing
`--sigma X` restricts the run to one noise rate. Use `--help` to see the
remaining reproducibility and accuracy options. The four-qubit executable
writes `results/four_qubit_noise_sweep.csv`.
