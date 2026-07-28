# Learning Parametrized Hamiltonians

Small, reproducible C++ benchmarks for learning a time-dependent Hamiltonian
from noisy queries to its unitary dynamics.

## First baseline

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
$M\in\{3,5,8,16\}$.

The baseline:

1. interpolates every matrix entry of $\widetilde U(t)$ in a degree-$M-1$
   Chebyshev series;
2. differentiates the series analytically;
3. forms $i\widehat U'(t)\widehat U(t)^\dagger$, then projects it onto the
   Hermitian, traceless matrices.

No projection of the noisy samples or interpolant onto the unitary group is
performed. This is intentionally the simplest unconstrained baseline.

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

## Build and run

The implementation is dependency-free apart from a C++20 compiler and CMake.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/lph_benchmark
```

The default run writes `results/chebyshev_baseline.csv`. Use `--help` to see
the available reproducibility and accuracy options.
