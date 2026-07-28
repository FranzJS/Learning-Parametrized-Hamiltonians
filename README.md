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

The default $\sigma=0.005$ gives 1% RMS relative Frobenius noise for a
two-qubit unitary. The seed is fixed to `20260727`, and the program reports
$M\in\{3,5,8,12,16\}$.

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

Accuracy for seed `20260727` and $\sigma=0.005$:

| $M$ | Unconstrained $E_H$ | Polar-projected $E_H$ |
|---:|---:|---:|
| 3 | 0.30682 | 0.30715 |
| 5 | 0.14956 | 0.14998 |
| 8 | 0.05598 | 0.05535 |
| 12 | 0.09448 | 0.09325 |
| 16 | 0.11693 | 0.11424 |

Intrinsic learner-construction runtime on the benchmark machine:

| $M$ | Unconstrained [$\mu$s] | Polar-projected [$\mu$s] |
|---:|---:|---:|
| 3 | 0.509 | 7.400 |
| 5 | 1.262 | 12.938 |
| 8 | 3.018 | 21.594 |
| 12 | 6.477 | 34.068 |
| 16 | 11.287 | 48.761 |

At $M=3,5$, interpolation bias dominates and the projection is slightly
counterproductive for this realization. It becomes beneficial once noisy
higher Chebyshev modes matter. At $M=16$, it lowers the error by about 2.3%,
while remaining negligible in absolute runtime.

## Build and run

The implementation is dependency-free apart from a C++20 compiler and CMake.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/lph_benchmark
```

The default run writes `results/chebyshev_unitary_projection.csv`. Use
`--help` to see the available reproducibility and accuracy options.
