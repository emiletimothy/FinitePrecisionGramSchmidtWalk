# Forward-error derivations for the three step-direction implementations

This note gives the results to cite for the step-direction error of the three
implementations in `gsw_stability/walks.py`. Each step says:

- **Cite**: the result that is used as a black box;
- **Statement**: the result itself;
- **Apply**: how it is used here.

The final bounds are first order in the unit roundoff. They assume the hypotheses
of each cited result, for example κε < 1, or that a factorization runs to
completion.

**Status tags**

| Tag | Meaning |
|---|---|
| **[verified]** | theorem number and form confirmed through papers that cite it |
| **[number unverified]** | standard result, but the theorem number is from memory |
| **[paraphrase]** | summarized from the abstract or from memory; check the source before quoting it |
| **[derived here]** | elementary, with the argument given in full |

## Notation

- $A$ is the alive set and $p$ the pivot. $S = A\setminus\{p\}$ is the set of
  non-pivot alive coordinates.
- The step direction solves $\min_x \|B_S x - b_p\|_2$. Then $u_S = -x$, $u_p = 1$
  and $u_i = 0$ off $A$.
- The residual $r = b_p - B_S x = Bu$ is the step in discrepancy space.
- $\kappa = \kappa_2(B_S) = \sigma_{\max}(B_S)/\sigma_{\min}(B_S)$.
- $\varepsilon$ is the unit roundoff: $2^{-53}$, $2^{-24}$ and $2^{-11}$ for float64,
  float32 and float16. Higham writes $u$ for it; here $u$ is the step direction.
- $\gamma_k = k\varepsilon/(1-k\varepsilon)$ and $\tilde\gamma_k = ck\varepsilon/(1-ck\varepsilon)$
  for a small constant $c$.
- Columns of $B$ have norm at most 1, so $\|B\|_2 \le \|B\|_F \le \sqrt{n}$.
- `tree_sum` uses pairwise summation. This only improves constants: the error
  constant of an inner product of length $k$ drops from $\gamma_k$ to about
  $\gamma_{\lceil \log_2 k\rceil + 1}$ (Higham, ch. 3–4). Every bound below keeps
  its form.

"Higham" means N. J. Higham, *Accuracy and Stability of Numerical Algorithms*,
2nd ed., SIAM, 2002.

---

## 1. `walk_lstsq`: least squares by Householder QR

### 1.1 The computed QR factor is backward stable

**Cite:** Higham, Theorem 19.4. **[verified]**

**Statement (Higham Thm 19.4).** Let $\hat R$ be the computed upper trapezoidal QR
factor of $A \in \mathbb{R}^{m\times n}$ ($m \ge n$) from Householder QR. Then
there is an orthogonal $Q$ with $A + \Delta A = Q\hat R$, where
$\|\Delta a_j\|_2 \le \tilde\gamma_{mn}\|a_j\|_2$ for $j = 1,\dots,n$. Here $Q$ is
the product of the Householder matrices that exact arithmetic would apply at each
step.

### 1.2 The computed least-squares solution is backward stable

**Cite:** Higham, Theorem 20.3. **[verified]**

**Statement (Higham Thm 20.3).** Let $A \in \mathbb{R}^{m\times n}$ ($m\ge n$) have
full rank, and suppose $\min_x\|b - Ax\|_2$ is solved by Householder QR. The
computed $\hat x$ is the exact least-squares solution of
$\min_x \|(b+\Delta b) - (A+\Delta A)x\|_2$, where

$$\|\Delta a_j\|_2 \le \tilde\gamma_{mn}\|a_j\|_2 \quad (j=1,\dots,n), \qquad \|\Delta b\|_2 \le \tilde\gamma_{mn}\|b\|_2 .$$

**Apply.** Take $A = B_S$ and $b = b_p$. The bound holds column by column. For a
single normwise bound, $\|\Delta B_S\|_2 \le \|\Delta B_S\|_F \le \tilde\gamma_{mn}\|B_S\|_F \le \tilde\gamma_{mn}\sqrt{n}\,\|B_S\|_2$.
So the backward error is $\eta := \sqrt{n}\,\tilde\gamma_{mn}$ in the normwise
sense. Column by column it is $\tilde\gamma_{mn}$, since every column has norm at
most 1.

### 1.3 Backward error → coefficient error

**Cite:** Higham, Theorem 20.1 (Wedin). Original: P.-Å. Wedin, *BIT* 13 (1973)
217–232. **[verified]**

**Statement (Higham Thm 20.1).** Let $A\in\mathbb{R}^{m\times n}$ ($m\ge n$) and
$A+\Delta A$ both have full rank. Let $x$ minimize $\|b - Ax\|_2$ with $r = b-Ax$,
and let $y$ minimize $\|(b+\Delta b)-(A+\Delta A)y\|_2$ with
$s = b+\Delta b-(A+\Delta A)y$. Suppose $\|\Delta A\|_2\le\epsilon\|A\|_2$ and
$\|\Delta b\|_2\le\epsilon\|b\|_2$. Then, provided $\kappa_2(A)\,\epsilon < 1$,

$$\frac{\|x-y\|_2}{\|x\|_2} \le \frac{\kappa_2(A)\,\epsilon}{1-\kappa_2(A)\,\epsilon}\left(2 + (\kappa_2(A)+1)\frac{\|r\|_2}{\|A\|_2\|x\|_2}\right),
\qquad
\frac{\|r-s\|_2}{\|b\|_2} \le (1+2\kappa_2(A))\,\epsilon .$$

**Apply.** With $\epsilon = \eta$ from 1.2, and to first order,

$$\|\Delta u\|_2 = \|\hat x - x\|_2 \lesssim \eta\left(2\kappa\|u_S\|_2 + \kappa^2\frac{\|Bu\|_2}{\|B_S\|_2}\right).$$

The residual bound in Thm 20.1 concerns $s$, the residual of the *perturbed*
problem. What we measure is $B_S(\hat x - x)$, and that bound is too crude to
show the observed flat behavior. So 1.4 derives the discrepancy-space error
directly.

### 1.4 Discrepancy-space error

**Cite:** first-order least-squares perturbation. It goes back to G. H. Golub and
J. H. Wilkinson, *Numer. Math.* 9 (1966) 139–148, and is in Golub–Van Loan, §5.3.
**[derived here]**

**Derivation.** Differentiate the normal equations $A^TAx = A^Tb$:

$$A^TA\,\delta x = A^T(\delta b - \delta A\,x) + \delta A^T r .$$

Hence

$$\delta x = A^+(\delta b - \delta A\,x) + (A^TA)^{-1}\delta A^T r,
\qquad
A\,\delta x = P_A(\delta b - \delta A\,x) + A(A^TA)^{-1}\delta A^T r,$$

where $P_A = AA^+$ is the orthogonal projector onto $\operatorname{range}(A)$,
$\|A^+\|_2 = 1/\sigma_{\min}$, $\|(A^TA)^{-1}\|_2 = 1/\sigma_{\min}^2$ and
$\|A(A^TA)^{-1}\|_2 = 1/\sigma_{\min}$.

**Apply.** Use the column-by-column perturbations from 1.2. Then
$\|\delta A\,x\|_2 \le \sum_j |x_j|\,\|\delta a_j\|_2 \le \tilde\gamma_{mn}\|x\|_1$,
because every column has norm at most 1. This gives

$$\|B\,\Delta u\|_2 = \|B_S\,\delta x\|_2 \lesssim \tilde\gamma_{mn}\left(\|b_p\|_2 + \|u_S\|_1\right) + \eta\,\kappa\,\|Bu\|_2 ,$$

$$\|\Delta u\|_2 = \|\delta x\|_2 \lesssim \frac{\tilde\gamma_{mn}(\|b_p\|_2 + \|u_S\|_1)}{\sigma_{\min}(B_S)} + \eta\,\frac{\|Bu\|_2}{\sigma_{\min}(B_S)^2} .$$

### 1.5 Specializing to the walk

**[derived here]** Since $u_p = 1$ and $Bu = B_A u_A$,
$\|Bu\|_2 \ge \sigma_{\min}(B_A)\,\|u_A\|_2 \ge \sigma_{\min}(B_A)$. For square or
near-square $B$, $\|Bu\|_2$ is typically of the order of $\sigma_{\min}(B_A)$. So
$\kappa\|Bu\|_2$ is about $\sigma_{\max}$ and $\|Bu\|_2/\sigma_{\min}^2$ is about
$\kappa/\sigma_{\max}$. Therefore:

- **Discrepancy space:** $\|B\,\Delta u\|$ is of order $\varepsilon(1 + \|u\|_1)$,
  essentially independent of κ while $\|u\|$ stays bounded. The measured slope is
  0.08.
- **Coefficients:** $\|\Delta u\|$ is of order $\varepsilon\,\kappa\,(1+\|u\|_1)$.
  The measured slope is 0.91.
- **When $\|u\|$ is large:** for near-collinear pairs, $\|u\|\sim 1/\eta_{\text{pair}}$
  while $Bu$ stays small, and the $\varepsilon\|u\|_1$ term makes the
  discrepancy-space error grow.

---

## 2. `walk_harshaw_cholesky`: Cholesky with the Woodbury identity (GSWDesign.jl)

Notation for this section:

- $X\in\mathbb{R}^{n\times d}$ has rows $x_i^T$ scaled so the largest row norm is 1.
- $B = [\sqrt\varphi\, I_n;\ \sqrt{1-\varphi}\,X^T]$.
- $Y = X_S \in \mathbb{R}^{|S|\times d}$ holds the rows of $X$ for units in $S$.
- $c = \varphi/(1-\varphi)$ and $M = cI_d + Y^TY$.
- $\sigma_1 \ge \dots$ are the singular values of $Y$, with $\sigma_k := 0$ for
  $k > \min(|S|, d)$.

### 2.1 The step direction equals the Woodbury form exactly

**Cite:** the push-through / Woodbury identity. W. W. Hager, "Updating the
inverse of a matrix", *SIAM Review* 31 (1989) 221–239; H. V. Henderson and
S. R. Searle, *SIAM Review* 23 (1981) 53–60. **[derived here]**

**Statement (Woodbury).** If $A$ and $C$ are invertible, then
$(A + UCV)^{-1} = A^{-1} - A^{-1}U(C^{-1} + VA^{-1}U)^{-1}VA^{-1}$.

**Apply.** $B_S^TB_S = \varphi I + (1-\varphi)YY^T$, and
$B_S^T b_p = (1-\varphi)\,Y x_p$, because $p\notin S$. The identity
$(\varphi I + (1-\varphi)YY^T)\,Y = (1-\varphi)\,Y M$ gives

$$u_S = -(B_S^TB_S)^{-1}B_S^Tb_p = -\,Y M^{-1} x_p .$$

### 2.2 Conditioning of $M$ against $B_S$

**Cite:** the nonzero eigenvalues of $YY^T$ and $Y^TY$ coincide. This is standard;
see e.g. Horn & Johnson, *Matrix Analysis*, §1.3. **[derived here]**

**Apply.**

$$\kappa(M) = \frac{\varphi + (1-\varphi)\sigma_1^2}{\varphi + (1-\varphi)\sigma_d^2},
\qquad
\kappa(B_S)^2 = \frac{\varphi + (1-\varphi)\sigma_1^2}{\varphi + (1-\varphi)\sigma_{|S|}^2}.$$

- $\kappa(M) \le \kappa(B_S)^2$ when $|S| \ge d$.
- $\kappa(M) \ge \kappa(B_S)^2$ when $|S| < d$, which happens late in the walk.
- Both are at most $1 + \sigma_1^2/c$, and $\|M^{-1}\|_2 \le 1/c$ always.

Two norms used below:

- $\|Y M^{-1}\|_2 = \max_k \sigma_k/(c+\sigma_k^2) \le 1/(2\sqrt c)$.
- $\|B_S\, Y M^{-1}\|_2 = \max_k \sqrt{1-\varphi}\,\sigma_k/\sqrt{c+\sigma_k^2} \le 1$,
  because $(B_SY)^T(B_SY) = (1-\varphi)\,Y^TY M$.

### 2.3 Backward error of the factor and of the solves

**Cite:** Higham, Theorem 10.3 **[verified]**, and the Cholesky-solve theorem that
follows it in §10.1 (Thm 10.4) **[number unverified]**. Triangular solves:
Higham Thm 8.5 **[number unverified]**.

**Statement (Higham Thm 10.3).** If Cholesky factorization is applied to the
symmetric positive definite $A\in\mathbb{R}^{n\times n}$ and runs to completion,
then the computed factor $\hat R$ satisfies $\hat R^T\hat R = A + \Delta A$ with
$|\Delta A| \le \gamma_{n+1}\,|\hat R^T||\hat R|$.

**Statement (Cholesky solve, Higham §10.1).** The computed solution of $Ax=b$ via
Cholesky satisfies $(A+\Delta A)\hat x = b$ with
$|\Delta A| \le \gamma_{3n+1}\,|\hat R^T||\hat R|$.

**Statement (triangular solve, Higham ch. 8).** The computed solution of $Tx = b$
by substitution satisfies $(T+\Delta T)\hat x = b$ with $|\Delta T| \le \gamma_n|T|$.

**Apply.** Here the factor is not a fresh Cholesky factorization. It is built by
one rank-one update per unit, then downdated at every step, so its backward error
accumulates. Write $\hat U^T\hat U = M + \Delta M_t$ with
$\|\Delta M_t\|_2 \le \delta(t)\,\|M\|_2$. Here $\delta(t)$ is exactly the drift
reported in the `max_drift` column and the `factor_drift_phi*.png` figures.
Thm 10.3 gives $\delta$ right after a refactor, with
$\delta \approx \gamma_{d+1}\,\| |\hat U^T||\hat U| \|_2/\|M\|_2 \lesssim d\,\gamma_{d+1}$.
The two triangular solves add a backward error of order $d\,\varepsilon\,\|M\|_2$.
In total, the solve is backward stable with $\eta_t \approx \delta(t) + c\,d\,\varepsilon$.

### 2.4 Backward error → forward error of $y = M^{-1}x_p$

**Cite:** Higham §7.1, the normwise perturbation theorem for linear systems
(Thm 7.2). **[number unverified]**

**Statement (normwise linear-system perturbation).** Let $Ax = b$ and
$(A+\Delta A)y = b+\Delta b$ with $\|\Delta A\|\le\epsilon\|E\|$ and
$\|\Delta b\|\le\epsilon\|f\|$, and assume $\epsilon\|A^{-1}\|\|E\| < 1$. Then

$$\frac{\|x-y\|}{\|x\|} \le \frac{\epsilon}{1-\epsilon\|A^{-1}\|\|E\|}\left(\frac{\|A^{-1}\|\|f\|}{\|x\|} + \|A^{-1}\|\|E\|\right),$$

and the bound is attainable to first order.

**Apply.** With $E = M$ and $f = 0$: $\|\Delta y\|/\|y\| \lesssim \eta_t\,\kappa(M)$.
To first order the error is structured, $\Delta y \approx -M^{-1}\Delta M_t\,y$.
This matters in 2.5.

### 2.5 From $y$ to the step direction, with an exact $M^{-1}x_p$ formula

**[derived here]** The step is $u_S = -Yy$, so $\Delta u_S = -Y\Delta y$. With the
structured $\Delta y \approx -M^{-1}\Delta M_t y$ and the norms from 2.2:

$$\|B\,\Delta u\|_2 = \|B_S Y M^{-1}\Delta M_t\, y\|_2 \le \eta_t\,\|M\|_2\,\|M^{-1}x_p\|_2 \le \eta_t\,\kappa(M)\,\|x_p\|_2,$$

$$\|\Delta u\|_2 \le \frac{\eta_t\,\|M\|_2\,\|M^{-1}x_p\|_2}{2\sqrt c}.$$

### 2.6 The formula GSWDesign.jl actually uses

**[derived here]** `compute_step_direction` does not compute $M^{-1}x_p$
directly. It computes $a = M^{-1}(L(Ux_p) - c\,x_p) = x_p - c\,y$, then
$-y = (a - x_p)/c$.

1. Forming the right-hand side through the factors costs
   $\|\Delta\mathrm{rhs}\| \lesssim d\,\varepsilon\,\|M\|_2\|x_p\|_2$.
2. The solve gives $\Delta a \approx M^{-1}(\Delta\mathrm{rhs} - \Delta M_t\, a)$,
   with $\|a\|_2 \le 2\|x_p\|_2$.
3. The subtraction $a - x_p$ cancels, and dividing by $c$ multiplies the
   remaining error.

Together these give

$$\|B\,\Delta u\|_2 \lesssim (d\,\varepsilon + 2\eta_t)\,\frac{\|M\|_2}{c}\,\|x_p\|_2,
\qquad
\|\Delta u\|_2 \lesssim (d\,\varepsilon + 2\eta_t)\,\frac{\|M\|_2}{c}\,\frac{\|x_p\|_2}{2\sqrt c},
\qquad
\frac{\|M\|_2}{c} = 1 + \frac{\sigma_1^2}{c}.$$

Compare 2.5, where the factor is $\|M\|_2\|M^{-1}x_p\|_2$, at most
$\kappa(M)\|x_p\|$ and often much less. The Julia formula always pays
$\|M\|/c \approx 1 + (1-\varphi)\sigma_1^2/\varphi$, which is large for small φ.

### 2.7 How $\delta(t)$ grows: the rank-one downdates

**Cite (the algorithm is stable):** A. W. Bojanczyk, R. P. Brent, P. Van Dooren
and F. R. de Hoog, "A note on downdating the Cholesky factorization",
*SIAM J. Sci. Stat. Comput.* 8 (1987) 210–221. **[paraphrase]**

**Statement.** The paper analyzes and compares three algorithms for downdating a
Cholesky factorization. Two are stable in a "mixed" sense: the computed
downdated factor is close to the exact downdate of slightly perturbed data. The
third is unstable.

Julia's `lowrankdowndate!` computes, row by row: $s = x_i/U_{ii}$,
$c_i = \sqrt{1-s^2}$, $U_{ij} \leftarrow (U_{ij} - s\,x_j)/c_i$, then
$x_j \leftarrow c_i x_j - s\,U_{ij}$, using the **new** $U_{ij}$. As far as I can
tell, this matches the paper's "mixed" hyperbolic-rotation form. I have not
confirmed it against the paper.

**Cite (sensitivity of the downdated factor):**
- G. W. Stewart, "The effects of rounding error on an algorithm for downdating a
  Cholesky factorization", *IMA J. Appl. Math.* 23 (1979) 203–213. **[paraphrase]**
  The LINPACK-type algorithm is stable in the presence of rounding error, but the
  downdated factor $\tilde R$ can be a very ill-conditioned function of $R$ and $x$.
- C.-T. Pan, "A perturbation analysis of the problem of downdating a Cholesky
  factorization", *Linear Algebra Appl.* 183 (1993) 103–116. **[paraphrase]**
  First-order bounds on the change in $\tilde R$ from perturbations of $R$ and $x$.
  They grow as $R^TR - xx^T$ approaches singularity.
- J.-G. Sun, "Perturbation analysis of the Cholesky downdating and QR updating
  problems", *SIAM J. Matrix Anal. Appl.* 16 (1995) 760–775. **[paraphrase]**
  Sharper perturbation bounds for the same problem.

**Apply.**
- **Per downdate:** mixed stability plus the perturbation bound give a
  forward-error increment of about $\varepsilon$ times the downdating condition
  number.
- **Accumulated drift:** summing the increments over the walk gives $\delta(t)$.
  The condition number grows as the downdated matrix $M - x_ix_i^T$ approaches
  singularity.
- **Small φ:** here $\lambda_{\min}(M) \ge c$ and $c$ is small, so the condition
  number is large and $\delta(t)$ grows quickly, as measured in float16.
- **Breakdown:** a run fails when the computed $s^2 \ge 1$, which in exact
  arithmetic would mean $M - x_ix_i^T$ is not positive definite.
- **Refactoring:** `refactor_every = k` resets $\delta$ to the Thm 10.3 level
  every k steps. It does not remove the $\|M\|/c$ factor from 2.6.

---

## 3. `walk_gs_compress`: explicit inverse of the Gram matrix (Low-Rank Thinning)

The walk is `kernel_gs_walk_cubic` from A. M. Carrell, A. Gong, A. Shetty,
R. Dwivedi and L. Mackey, "Low-Rank Thinning", ICML 2025, arXiv:2502.12063,
App. B.6. It forms $Q = B^TB$, sets $C = (Q_{S,S})^{-1}$, computes
$u_S = -C\,Q_{S,p}$, and removes indices from $C$ by block inversion plus
Sherman–Morrison.

### 3.1 Forming $Q$ already costs κ²

**Cite:** Higham §3.5, error in matrix multiplication. **[number unverified]**

**Statement.** The computed product satisfies $|\widehat{AB} - AB| \le \gamma_k\,|A||B|$,
where $k$ is the inner dimension. With pairwise summation, $\gamma_k$ becomes
about $\gamma_{\lceil\log_2 k\rceil+1}$.

**Apply.** $|\Delta Q| \le \gamma_m |B|^T|B|$, so
$\|\Delta Q\|_2 \le \gamma_m\,\|\,|B|\,\|_F^2 \le \gamma_m\, n$. By Weyl's
inequality, $\lambda_{\min}(\hat Q_{S,S}) \ge \sigma_{\min}(B_S)^2 - \|\Delta Q\|_2$.
Everything in $Q$ below about $\gamma_m n$ is lost. The factorization can break
down once $\sigma_{\min}(B_S)^2 \lesssim \gamma_m n$, that is, once
$\varepsilon\,\kappa^2 \gtrsim 1$ for $\sigma_{\max}$ of order 1. This is where the
float16 and float32 runs began to fail.

### 3.2 The explicit inverse

**Cite:** J. J. Du Croz and N. J. Higham, "Stability of methods for matrix
inversion", *IMA J. Numer. Anal.* 12 (1992) 1–19 **[verified: paper; statement
paraphrased]**. Also Higham ch. 14.

**Statement.** Unblocked methods for inverting a triangular matrix $T$ satisfy
small componentwise bounds on either the left or the right residual, of the form
$|\hat X T - I| \le c_n\varepsilon\,|\hat X||T|$ (or $|T\hat X - I| \le \dots$).
Which residual is small depends on the loop ordering. Consequently the forward
error is of order $c_n\varepsilon\,\kappa$.

**Apply.** Here $C = \hat R^{-1}\hat R^{-T}$ comes from the Cholesky factor of
$\hat Q_{S,S}$. The inverse also carries the perturbation from 3.1, and
$(Q+\Delta Q)^{-1} = Q^{-1} - Q^{-1}\Delta Q\,Q^{-1} + O(\|\Delta Q\|^2)$.
Together,

$$\frac{\|\hat C - Q_{S,S}^{-1}\|_2}{\|Q_{S,S}^{-1}\|_2} \lesssim c\,\varepsilon\,\kappa(Q_{S,S}) = c\,\varepsilon\,\kappa^2 .$$

### 3.3 Solving with the explicit inverse

**Cite:** A. Druinsky and S. Toledo, "How accurate is inv(A)*b?",
arXiv:1201.6035 (2012). **[verified: abstract]**

**Statement.** "Under reasonable assumptions on how the inverse is computed,
x = inv(A)*b is as accurate as the solution computed by the best backward-stable
solvers." The assumptions concern the residual of the computed inverse; bounds
like those in 3.2 are the kind that supply them. This is a forward-error result;
it says nothing about the error measured through $B$.

**Apply.** For the coefficients, $\|\Delta u\|_2 \lesssim c\,\varepsilon\,\kappa^2(\|u_S\|_2 + 1)$.
For the discrepancy space **[derived here]**:

- **Structured error.** If the error is of the form $\Delta C = -C\,\Delta Q\,C$,
  as in 3.2, then $\|B_S\Delta C\,q\| \le \|B_SC\|\,\|\Delta Q\|\,\|Cq\| = \|\Delta Q\|\,\|x\|/\sigma_{\min}$,
  because $B_SC = B_S(B_S^TB_S)^{-1}$ has norm $1/\sigma_{\min}$. That is of order
  $\varepsilon\,\kappa\,\|u\|$.
- **Unstructured error**, with $\|\Delta C\| \le \epsilon_C\|C\|$:
  $\|B_S\Delta C\, q\| \le \sigma_{\max}\,\epsilon_C\,\|C\|\,\|q\| \lesssim \epsilon_C\,\kappa^2$.

So the discrepancy-space error lies between about $\varepsilon\kappa$ and about
$\varepsilon\kappa^2$. The measured slopes are 1.2 (float16) and 1.9 (float32).

### 3.4 Each Sherman–Morrison removal

**Cite:** L. Ma, C. Boutsikas, M. Ghadiri and P. Drineas, "A Note on the
Stability of the Sherman–Morrison–Woodbury Formula", arXiv:2504.04554 (2025),
Theorem 2. **[verified: statement from the paper]**

**Statement (Theorem 2).** Let $B = A + UV^T$. Suppose approximate inverses of
$A$ and of the capacitance matrix $Z = I + V^TA^{-1}U$ are available, with errors
$\epsilon_1$ and $\epsilon_2$ in the 2-norm. Let $\tilde B^{-1}$ be the
Sherman–Morrison–Woodbury formula evaluated with them. Then

$$\|B^{-1} - \tilde B^{-1}\|_2 \le \epsilon_1 + \epsilon_1\lambda\alpha\left(2\|A^{-1}\|_2 + \epsilon_1\right) + \lambda\left(\|A^{-1}\|_2 + \epsilon_1\right)^2\left(\epsilon_2 + 2\epsilon_1\lambda\alpha^2\right),$$

with $\alpha = \|(I + V^TA^{-1}U)^{-1}\|_2$ and $\lambda = \|U\|_2\|V\|_2$. Check
the paper for the exact definition of the approximate capacitance inverse.

**Apply. [derived here]** Removing index $i$ updates
$C_{\text{new}} = D - Dqq^TD/(Q_{ii} + q^TDq)$, where $D$ is the old inverse
restricted to the remaining indices and $q = Q_{S',i}$. This is exactly
Sherman–Morrison, $(A + UV^T)^{-1}$, with $A = D^{-1}$, $U = q/Q_{ii}$ and $V = q$.
Here $A = D^{-1}$ is the Schur complement $Q_{S'} - qq^T/Q_{ii}$, and
$A + UV^T = Q_{S'}$. Then:

- **Capacitance:** $Z = 1 + q^TDq/Q_{ii} \ge 1$, because $D$ is SPD, so
  $\alpha \le 1$. The Sherman–Morrison denominator never amplifies the error.
- **Growth factor:** $\lambda\|A^{-1}\|_2 = \|q\|_2^2\|D\|_2/Q_{ii}$, where
  $\|D\| \approx 1/\sigma_{\min}^2$.
- **Recurrence:** with $\epsilon_1$ the error already carried by $D$,
  $\epsilon_{\text{new}} \lesssim \epsilon_1\left(1 + 2\|q\|^2\|D\|/Q_{ii}\right) + (\|q\|^2/Q_{ii})\,\|D\|^2\,\epsilon_2$.
  Iterating it over the removals bounds the drift $\|CQ - I\|$ that
  `factor_drift_phi*.png` shows.

### 3.5 Bounds covering every rounding error, and classic SMW stability

**Cite:** B. Hashemi and Y. Nakatsukasa, "Error bounds for the Sherman–Morrison
formula and its modification with improved stability", arXiv:2609.12266
(Sept. 2026). **[paraphrase of the abstract]**

**Statement.** The paper derives a backward error bound for the Sherman–Morrison
formula that accounts for every rounding error and needs no conditions on the
size of the capacitance. It also gives a forward error bound, with computable
growth factors that can certify stability on a given problem. It states that the
Sherman–Morrison formula is not numerically stable in general, and proposes a
modification with built-in self-correction, observed to be backward stable.

**Cite:** E. L. Yip, "A note on the stability of solving a rank-p modification of
a linear system by the Sherman–Morrison–Woodbury formula", *SIAM J. Sci. Stat.
Comput.* 7 (1986) 507–513. **[paraphrase]** A stability analysis of
Sherman–Morrison–Woodbury for rank-p updates, with bounds on the condition number
of the capacitance matrix.

---

## Summary of the predicted scaling

These are the expected orders of magnitude, dropping constants polynomial in n.

| Implementation | Coefficient error ‖Δu‖ | Discrepancy error ‖BΔu‖ | Main results used |
|---|---|---|---|
| `walk_lstsq` (Householder QR) | ε(κ‖u‖ + κ²‖Bu‖) → ~εκ in the walk | ε(1 + ‖u‖₁ + κ‖Bu‖) → ~ε in the walk | Higham Thms 19.4, 20.3, 20.1; §1.4 |
| `walk_harshaw_cholesky` | (dε + η_t)·(‖M‖/c)·‖x_p‖/(2√c) | (dε + η_t)·(‖M‖/c)·‖x_p‖ | Higham Thm 10.3, §7.1; Bojanczyk et al.; Pan; Sun |
| `walk_gs_compress` | εκ²(‖u‖ + 1) | between εκ‖u‖ and εκ² | Higham §3.5; Du Croz–Higham; Druinsky–Toledo; Ma et al. Thm 2 |

Here η_t = δ(t) + O(dε) is the backward error of the factor at step t, and
‖M‖/c = 1 + σ₁(X_S)²/c.

## Links

- [Higham, *Accuracy and Stability*, ch. 20 (SIAM)](https://epubs.siam.org/doi/abs/10.1137/1.9780898718027.ch20)
- [Epperly, Meier & Nakatsukasa: fast randomized LS solvers (cites Thms 20.1 and 20.3)](https://arxiv.org/pdf/2406.03468)
- [Probabilistic rounding error analysis of Householder QR (cites Thm 19.4)](https://eprints.maths.manchester.ac.uk/2865/1/paper.pdf)
- [Higham, "Cholesky factorization" (MIMS EPrint)](https://eprints.maths.manchester.ac.uk/1199/1/chol08.pdf)
- [Du Croz & Higham 1992 (PDF)](https://nhigham.com/wp-content/uploads/2023/09/duhi92.pdf)
- [Druinsky & Toledo, "How accurate is inv(A)*b?"](https://arxiv.org/abs/1201.6035)
- [Ma, Boutsikas, Ghadiri & Drineas 2025](https://arxiv.org/html/2504.04554)
- [Hashemi & Nakatsukasa 2026](https://arxiv.org/abs/2609.12266)
- [Bojanczyk, Brent, Van Dooren & de Hoog 1987 (SIAM)](https://epubs.siam.org/doi/10.1137/0908031)
- [Bojanczyk et al. 1987 (Brent's PDF)](https://maths-people.anu.edu.au/~brent/pd/rpb095a.pdf)
- [Stewart 1979](https://academic.oup.com/imamat/article-abstract/23/2/203/657384)
- [Pan 1993](https://www.sciencedirect.com/science/article/pii/002437959390426O)
- [Yip 1986 (ProQuest)](https://www.proquest.com/openview/9d6b8bfce977122c82fd90c981f905cd/1.pdf?pq-origsite=gscholar&cbl=666298)
- [Carrell et al., Low-Rank Thinning](https://arxiv.org/abs/2502.12063)
- [Harshaw et al., Gram–Schmidt walk design](https://arxiv.org/abs/1911.03071)
