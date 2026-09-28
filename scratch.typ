#let inner(l, r) = $chevron.l #l, #r chevron.r$
= Setup
We massage the update rule as follows
$ (delta_t u_t + epsilon_t) = (delta_t + epsilon_t (p))u_t - epsilon_t (p) u_t $
to get
#let dt = $delta_t + epsilon_t (p)$
$ B(z - z_0) = B sum_t (dt u_t + epsilon_t - epsilon_t (p)u_t) = 
\ B sum_t (dt u_t) + B sum_t epsilon_t - epsilon_t (p) u_t $
Let's focus on the first term and proceed under the assumption that that second part is bounded, collecting it later either using Orlicz norm stuff or just Cauchy Schwarz and absorbing it as a linear term in the mgf

// TODO: decide how we want to notate the pivot b_(pt), maybe we'd prefer consistency with BDGL
#let wr = $w_sigma(r)$
#let ar = $inner(wr, v)$
#let br = $inner(wr, b_(p t))$
#let update = $(dt inner(B u_t, v))$
#let sum2 = $sum_(s in S_p) sum_(r in Q_s)$
To show that $B sum_t (dt) u_t$ is $1$-subgaussian, we want $forall v in RR^d$ that we have
$ EE [exp(inner(B sum_t (dt) u_t, v))] <= exp(1/2 norm(v)^2) $
using linearity to rearrange the inner product on the left hand side, and moving the right hand side to the left, it suffices to show
$ EE [exp((sum_t (dt) inner(B u_t, v))-1/2 norm(v)^2)] <= 1 $
Because ${w_sigma(r)}$ forms an orthonormal basis (assuming B is full column rank) we can write $norm(v)^2 = sum_(r=1)^n inner(wr, v)^2$ so it suffices to show
$ EE [exp(sum_t update - 1/2 sum_r ar^2)] <= 1 $
Letting $Q_t$ denote the set of indices $r$ such that $b_r$ is decided in iteration $t$, we can group the summations over pivot phases as follows:
// $ EE [exp(sum_(S_p) (sum_(t in S_p) ((dt) inner(B u_t, v) - 1/2 sum_(r in Q_t) ar^2)))] <= 1 $
$ EE [exp(sum_(S_p) (sum_(t in S_p) update - 1/2 sum2 ar^2))] <= 1 $
it is therefore clear that it suffices to show for each pivot phase $S_p$, with $Delta_(S_p)$ denoting randomness up to $S_p$, that
$ EE [exp(sum_(t in S_p) (update - 1/2 sum2 ar^2)) bar Delta_(S_p)] <= 1 $
and noting that $norm(b_( p t)) <= 1$ implies $sum_(s in S_p) sum_(r in Q_s) br^2 <= 1$, it suffices to show the stronger inequality

// TODO: for measurability reasons that are not yet clear to me you need to put
// the "damping" inside the left hand side

$ EE [exp(sum_(t in S_p) update - 1/2 (sum2 br^2) (sum2 ar^2)) bar Delta_(S_p)] <= 1 $
We now reuse a lemma of Spielman which characterizes the ideal step direction $B u_t$, we have that
$ B u_t = sum_(s in S_p \ s < t) sum_(r in Q_s) br w_sigma(r) $
implying that
$ inner(B u_t, v) = sum_(s in S_p \ s < t) sum_(r in Q_s) ar br $
Defining $alpha_r = br, beta_r = ar $ and substituting the previous lemma gives
$ EE [exp(sum_(t in S_p) (dt) sum_(s in S_p \ s < t) sum_(r in Q_s) alpha_r beta_r - 1/2 (sum2 alpha_r^2) (sum2 beta_r^2)) bar Delta_(S_p)] <= 1 $
Now we're gonna define $g(R)$ and then we induct over $R$ to show the result, each induction step involves
conditioning on the iterations where $s<t$ wins this minimization for the first sum
#let sumR1 = $sum_(s in S_p \ s < t \ s <= R) sum_(r in Q_s)$
#let sumR2 = $sum_(s in S_p \ s <= R) sum_(r in Q_s)$
$ g(R) := EE [exp(sum_(t in S_p) (dt) sumR1 alpha_r beta_r - 1/2 (sumR2 alpha_r^2) (sumR2 beta_r^2)) bar Delta_(S_p)] <= 1 $
= The Lemma, Actual new stuff
To do the induction we're actually gonna condition on $Delta_R$ so that all the summations
match, I sure do hope that I can change this to address measurability later without too much work
$ EE [exp(sum_(t in S_p \ t > R) (dt) sumR2 alpha_r beta_r - 1/2 (sumR2 alpha_r^2) (sumR2 beta_r^2)) bar Delta_(R)] <= 1 $
Observe that defining $x = z_(R+1) (p),$ we have $sum_(t in S_p \ t > R) (dt) in {1 - x, -1 - x}$.
In the case where this random variable is mean zero we would take $1-x$ with probability $p = (1+x)/2$, however we assume
#let tx = $tilde(x)$
here we have no such assumption. We instead assume the true probability is some $(1 + tx)/2$ where $tx in [-1,1]$
and will defer justifying the assumption $abs(tx - x) <= gamma$. Anyways this lets us take the expectation:
$ EE [exp(sum_(t in S_p \ t > R) (dt) sumR2 alpha_r beta_r - 1/2 (sumR2 alpha_r^2) (sumR2 beta_r^2)) bar Delta_(R)] $
$ = (1 + tx)/2 exp((1 - x)eta_R - 1/2 a_R b_R) + (1 - tx)/2 exp((-1 - x)eta_R - 1/2 a_R b_R) $
$ = exp(tx - x) ((1 + tx)/2 exp((1 - tx)eta_R - 1/2 a_R b_R) + (1 - tx)/2 exp((-1 - tx)eta_R - 1/2 a_R b_R)) $
$ = exp(tx -x) f_tx mat(eta_R, a_R; b_R, eta_R) $


// $ bb(E) [sum_(t in S_p) (delta_t inner(B u_t, v)) - 1/2 norm(P_p b_(p_t))^2 norm(P_p v)^2 | Delta_p] <= 1 $
// applying the lemma about $B u_t$ and rewriting the projection in terms of the projection onto the appropriate gram schmidt vectors:
// $ arrow.l.r.long bb(E) [sum_(t in S_p) (delta_t inner(sum_(s in S_p \ s <= t) sum_(r in Q_s) inner(w_(sigma(r)), b_(p_t)) w_(sigma(r)), v)) - 1/2 (sum_(s in S_p) sum_(r in Q_s) inner(w_(sigma(r)), b_(p_t))^2) (sum_(s in S_p) sum_(r in Q_s) inner(w_(sigma(r)), v)^2) | Delta_(S_p) ] <= 1 $
// $ arrow.l.r.long bb(E) [sum_(t in S_p) delta_t (sum_(s in S_p \ s <= t) sum_(r in Q_s) inner(w_(sigma(r)), b_(p_t))  inner(w_(sigma(r)), v)) - 1/2 (sum_(s in S_p) sum_(r in Q_s) inner(w_(sigma(r)), b_(p_t))^2) (sum_(s in S_p) sum_(r in Q_s) inner(w_(sigma(r)), v)^2) | Delta_(S_p) ] <= 1 $
// Define $alpha_r := inner(w_(sigma(r)), b_(p_t))$ and $beta_r := inner(w_(sigma(r)), b_(p_t))$
// $ arrow.l.r.long bb(E) [sum_(t in S_p) delta_t (sum_(s in S_p \ s <= t) sum_(r in Q_s) alpha_r beta_r) - 1/2 (sum_(s in S_p) sum_(r in Q_s) alpha_r^2) (sum_(s in S_p) sum_(r in Q_s) beta_r^2) | Delta_(S_p) ] <= 1 $