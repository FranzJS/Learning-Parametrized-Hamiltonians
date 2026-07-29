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

Accuracy for seed `20260727`; each entry is
unconstrained / polar-projected:

| Relative noise | $M=3$ | $M=5$ | $M=8$ | $M=12$ | $M=16$ |
|---:|---:|---:|---:|---:|---:|
| 0.5% | .48909 / .48932 | .21093 / .21068 | .03141 / .03122 | .03183 / .03123 | .04315 / .04242 |
| 1% | .48891 / .48935 | .21184 / .21125 | .04503 / .04420 | .06360 / .06237 | .08630 / .08483 |
| 2% | .48874 / .48953 | .21520 / .21366 | .07859 / .07638 | .12719 / .12463 | .17261 / .16959 |
| 3% | .48882 / .48988 | .22052 / .21773 | .11436 / .11080 | .19082 / .18683 | .25897 / .25429 |

Intrinsic post-processing runtime in microseconds; each entry is
unconstrained / polar-projected:

| Relative noise | $M=3$ | $M=5$ | $M=8$ | $M=12$ | $M=16$ |
|---:|---:|---:|---:|---:|---:|
| 0.5% | 6.31 / 179.88 | 14.58 / 305.18 | 33.76 / 504.38 | 69.37 / 784.56 | 120.09 / 1079.98 |
| 1% | 6.31 / 179.93 | 14.62 / 305.13 | 33.83 / 504.55 | 69.43 / 783.77 | 120.60 / 1090.73 |
| 2% | 6.31 / 179.99 | 14.60 / 305.21 | 33.49 / 503.60 | 69.19 / 783.98 | 120.22 / 1079.01 |
| 3% | 6.31 / 223.35 | 14.59 / 350.00 | 33.82 / 577.01 | 69.10 / 870.70 | 119.72 / 1223.79 |

Synthetic data generation includes RK4 evolution to all query times and
addition of the Gaussian noise. It is shared by both algorithms and does not
depend materially on the noise rate:

| $M$ | Data generation [ms] |
|---:|---:|
| 3 | 147.85 |
| 5 | 151.10 |
| 8 | 156.74 |
| 12 | 157.47 |
| 16 | 156.56 |

The post-processing time excludes reconstruction on the quadrature nodes used
only to evaluate the error. The data-generation implementation exploits the
sparse Pauli form of $H_4(t)$; its absolute runtime is machine-dependent.
Halving the RK4 step from $2\cdot10^{-4}$ to $10^{-4}$ changes every reported
error by at most $1.34\cdot10^{-14}$.

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
