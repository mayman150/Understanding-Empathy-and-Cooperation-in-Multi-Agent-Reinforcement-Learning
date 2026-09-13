# Empathy from one's own value function: what the "value signal" does, what we measured, and how to fix it

*Working note, September 2026.  Statements about our own runs are measurements (8 fresh seeds, 95 %
confidence intervals unless stated otherwise).  Statements about the literature carry a citation.
Everything else is marked as a derivation or as a hypothesis with the experiment that would test it.*

---

## 1. Setting and notation

We work in a partially observable Markov game (stochastic game) with `N` agents (Shapley 1953; Littman
1994).  At step `t` agent `i` receives a private observation `o_i^t`, takes `a_i^t`, and receives a reward
`r_i^t` that depends on the joint action.  Each agent maximises its own discounted return
`J_i = E[ sum_t gamma^t r_i^t ]`.  Learning is fully decentralised: one independent PPO learner per agent
(Schulman et al. 2017; CleanRL implementation), with a critic `V_i` trained on agent `i`'s own returns from
agent `i`'s own observations, and the generalised advantage estimate `A_i^t` (Schulman et al. 2016).  In the
code (`train.py`) the advantage that enters the clipped surrogate is normalised per minibatch
(`norm_adv`), which matters below.

A *social preference* is a function `F_i(z)` of a vector `z` whose entry `z_ij` is "agent `i`'s estimate
of how well off agent `j` is".  With `zbar_i` the mean over `j != i` (`empathy_marl/empathy.py`):

| formulation | `F_i(z)` | origin |
|---|---|---|
| EI  | `alpha * zbar_i` | "empathetic influence": weight on the others' welfare |
| SVO | `alpha * (cos(phi) z_ii + sin(phi) zbar_i)` | social value orientation; the `cos/sin` form is the one used by Schwarting et al. (2019) and McKee et al. (2020) / Madhushani et al. (2023) |
| SIA | `-alpha * |z_ii - zbar_i|` | symmetric inequity aversion |
| IA  | `-alpha/(N-1) sum_j max(z_ij - z_ii, 0) - beta/(N-1) sum_j max(z_ii - z_ij, 0)` | Fehr & Schmidt (1999); Hughes et al. (2018) |

The two "signals" differ only in what `z` is and where `F_i` enters.

## 2. The two signals as implemented

### 2.1 Reward signal (the literature)

`z_ij = e_j^t`, the temporally smoothed reward of agent `j`, `e_j^t = gamma * lambda * e_j^{t-1} + r_j^t`, and
`F_i(e)` is an intrinsic reward added to `r_i` before the return is computed.  This is exactly the
construction of Hughes et al. (2018, eqs. 3-4), who write that they "allow agents to observe the smoothed
reward of every player on each timestep" and report their strongest results at `alpha = 5, beta = 0.05`
(with `lambda = 0.975`).  With `lambda = 0` and the SVO form it is the effective reward
`r_i cos(theta) + rbar_{-i} sin(theta)` of Schwarting et al. (2019) / McKee et al. (2020).  The reward
signal therefore requires access to the other agents' rewards.

### 2.2 Value signal (our proposal)

`z_ij = V_i(o_j^{t+1})`: agent `i`'s **own** critic evaluated on agent `j`'s next observation.  No agent
ever sees another agent's reward.  `X_i^t = F_i(z)` is computed once per rollout under `no_grad` and added
to the advantage: the PPO surrogate uses `A_i^t + X_i^t` (then normalised).  The term is masked at episode
ends.

The closest prior work is *Empathic DQN* (Bussmann, Heinerman & Lehman 2019), which also evaluates the
agent's own value function on a constructed state in which the two agents' positions are swapped, and
combines self-centred and other-centred value with a "selfishness" weight; they assume "some types of
reward ... may generalize across agents" and test in gridworlds against fixed co-players.  Raileanu et al.
(2018) use the agent's own *policy* to infer another agent's hidden goal (self other-modeling).  Our
setting differs in that both agents learn, the term enters a policy-gradient advantage rather than a
Q-target, and we compare against the reward-access baseline with the same functional forms.

## 3. What we measured

### 3.1 Repeated Prisoner's Dilemma (2 players, 5M steps, 2 seeds, SIA, value signal)

* Every configuration, including plain PPO, converged to mutual defection within about 300k steps
  (final cooperation 0.000, policy entropy 0).  Larger `alpha` made the collapse *faster*, monotonically.
* The measured gap `|V_0(o_0') - V_0(o_1')|` was 0.15 in the first 25k steps (untrained critic), 0.007 by
  50k, and numerically zero from 400k on.  Under any state-independent policy the PD's one-round
  observation carries no information about the future, so `V_i(o_i) = V_i(o_j)` exactly and every
  gap-based value term is identically zero; for EI/SVO the term is a constant that advantage
  normalisation removes.  (Derivation, confirmed by the measurement.)
* Independently of the signal, SIA's differential against an opponent that cooperates with probability
  `q` is `-alpha * g * (1 - 2q)` (with `g` the gap after a mismatched round): it favours matching the
  majority action, which is why it accelerates defection.  (Derivation; observed.)

### 3.2 Coin Game (Lerer & Peysakhovich 2017; rules as in LOLA, Foerster et al. 2018: +1 for any coin, the
owner loses 2 when the other agent takes a coin of the owner's colour)

Stage B, each method at its own best configuration selected on separate tuning seeds (see
`scripts/best_configs.py` for the protocol), 8 fresh seeds, 2M steps.  Final = mean over the last 200k
steps; `+/-` = half-width of the 95 % CI over seeds.  Cooperation rate = own-colour share of an agent's
pickups (0.5 = grabs everything, 1.0 = only its own colour), averaged over the two agents.

| method | collective return | cooperation rate | vs. own-value control |
|---|---|---|---|
| plain PPO | 0.00 +/- 0.04 | 0.500 +/- 0.000 | |
| **own-value control** (`alpha * V_i(o_i')`, no other-regarding term) | 15.60 +/- 2.20 | 0.760 +/- 0.058 | |
| EI, value signal | 19.56 +/- 1.03 | 0.802 +/- 0.009 | +4.0 (z = 3.9) / +0.04 (n.s.) |
| IA, value signal | 18.15 +/- 1.72 | 0.798 +/- 0.040 | +2.5 (z = 2.2) / +0.04 (n.s.) |
| SVO (phi = pi/6), value signal | 15.80 +/- 0.94 | 0.833 +/- 0.021 | +0.2 (n.s.) / +0.07 (z = 2.8) |
| SIA, value signal | 3.79 +/- 1.95 | 0.537 +/- 0.022 | |
| EI, reward signal (has reward access) | 33.27 +/- 0.17 | 0.996 +/- 0.001 | |
| IA, reward signal | 5.06 +/- 0.76 | 0.614 +/- 0.015 | |
| SIA, reward signal | 6.11 +/- 1.43 | 0.627 +/- 0.026 | |

Three facts to keep in mind for the analysis:

1. The value signal does produce cooperation without reward access (0 -> 16-20 collective return, 0.50 ->
   0.80 cooperation), robustly across formulations and seeds.
2. A term with **no** other-regarding component, `alpha * V_i(o_i')`, produces most of it (15.6 / 0.76).
   The other-regarding part adds a few points at most: significant for EI-value on collective return and
   for SVO-value on cooperation rate, not for the rest.
3. At every `alpha` that had an effect, the social term was 2 to 500 times the advantage
   (`social/term_abs_mean` vs `social/advantage_abs_mean`).

## 4. Analysis

### 4.1 What adding `X` to the advantage does (derivation)

The policy-gradient theorem gives `grad J = E[ grad log pi(a|s) * Q(s,a) ]` (Sutton et al. 2000).  Any
*baseline* `b(s)` that does not depend on the action can be subtracted without changing this expectation
(Williams 1992); a term that depends on the action, or on the next state, is not a baseline and changes
what is being optimised unless it is explicitly corrected for (see Tucker et al. 2018 for the discussion
of action-dependent baselines).  Our `X_i^t = F_i(V_i(o^{t+1}))` depends on `s^{t+1}` and hence on `a^t`.
Writing `Xbar(s, a) = E[X | s, a]`, the score-function identity gives

    E[ grad log pi(a|s) * X ] = grad_theta E_{a ~ pi(.|s)} [ Xbar(s, a) ].

So the modified estimator is the gradient of `J_i + alpha_eff * E_s E_{a~pi} [Xbar(s,a)]`: the agent
maximises its return **plus a one-step-ahead, undiscounted, non-bootstrapped preference for next states with
high `F_i(V_i(.))`**.  Nothing propagates this preference through time; it is myopic by construction.  This
is a correct description of the current method, not a criticism in itself, but it tells us what the
method is: greedy shaping towards states the critic rates highly, plus PPO.

### 4.2 Scale

Because `V` is on the return scale and `A` on the TD-error scale, and because the sum is normalised per
minibatch, at the `alpha` values that worked the normalised coefficient was dominated by `X`.  The
measured ratios (`|X|/|A|` from 2 to 500) mean the agents were, to a first approximation, optimising the
one-step preference of 4.1 with PPO's own advantage as a perturbation.  This is consistent with SVO-value
agents collecting fewer coins than the control (24 vs 30 per episode): the selfish learning signal is
attenuated.

### 4.3 Level versus change, and why "imagined value" is a poor reward proxy

A natural repair is to use the *change* of the other's prospects, `V_i(o_j^{t+1}) - V_i(o_j^t)`, as an
intrinsic reward.  Potential-based reward shaping (Ng, Harada & Russell 1999) shows why this cannot change
what is learned in the limit: for any function `Phi` of the state, adding `F = gamma Phi(s') - Phi(s)`
to the reward leaves the optimal policy unchanged, because the discounted sum of `F` along any trajectory
telescopes to `gamma^T Phi(s_T) - Phi(s_0)`, independent of the actions taken.  Devlin & Kudenko (2011)
extend this to multi-agent stochastic games: potential-based shaping does not alter the set of Nash
equilibria (Lu et al. 2011 prove the same for general-sum stochastic games), and Devlin & Kudenko (2012)
show it also holds for potentials that change over time, which covers a `Phi = V_i` that is still being
learned.  The "imagined reward" `V_i(o_j^t) - gamma V_i(o_j^{t+1})` is exactly such a term with
`Phi = -V_i(o_j)`.  Its total over an episode is fixed by the initial and terminal states; no policy of
agent `i` can increase it.

Two things follow.  (i) Any *value-difference* social reward is inert with respect to equilibria and can
only affect the learning dynamics.  (ii) Our *level* term `alpha * V_i(o_j')` is not potential-based, so
it does change the objective, but in the uncontrolled, myopic way described in 4.1.  The value function
is a forecast under the current joint policy; it is not a quantity whose sum over time measures how well
the other agent actually did.

### 4.4 What the control suggests (hypothesis)

Why would `alpha * V_i(o_i')`, which never looks at the other agent, produce cooperation?  In the Coin
Game the critic has reason to rate "the other agent's coin is on the board" above "my coin is on the
board": in the first state I cannot lose 2, in the second I can.  Taking my own coin respawns the coin in
the other colour (the safe state); taking the other's coin gives +1 now but respawns it in mine (the risky
state); leaving the other's coin keeps the safe state.  A return-maximiser takes the other's coin because
+1 now outweighs the state difference; an agent that is greedy for next-state value (4.1, 4.2) does not.
The observable consequence is "takes own coins, leaves the other's", i.e. cooperation, from
self-protection.  In numbers, with `S` the critic's board score (`S(A)` for "other's coin on the board",
`S(B)` for "my coin on the board") and Red next to the other's coin: stealing gives `1 + alpha * S(B)`,
leaving gives `alpha * S(A)`, so leaving wins whenever `alpha * (S(A) - S(B)) > 1`; the true return does
not show this difference because after leaving, the other agent takes the coin and the board becomes `B`
anyway.  This is a hypothesis.  It predicts (a) a probe of the trained critic shows `S(A) > S(B)`, and
(b) the effect depends on the term being a non-potential shaping: fed into GAE *as a reward*, the level
`alpha * V_i(o_i')` should still produce cooperation (it changes the objective), while the difference
`alpha * [gamma V_i(o_i') - V_i(o_i)]` should not (policy-invariant, Section 4.3).  Note that the
difference cannot be tested as an *advantage coefficient*: there `V_i(o_i)` is action-independent, acts as
a baseline, and the update is identical to the level version in expectation.

EI-value's trajectory is different from the control's (anti-social first: cooperation 0.29 and collective
return -12.5 at 70k steps; then a monotone climb to the best value-signal result with the tightest CI).
Scoring boards from Blue's seat with Red's critic makes the mechanism concrete.  Let "threat" = Blue's
coin on the board with Red adjacent (about to lose 2, score low), "safe" = a Red-colour coin on the board
(Blue cannot lose, score high), "own coin, Red far" = Blue's coin with Red away (score high).  At the
moment Red decides whether to steal, stealing turns "threat" into "safe": the -2 is a reward event that the
next observation does not show, while the respawned coin is Red's colour and therefore safe for Blue.  So
the level term `alpha * V_i(o_j')` *rewards the steal*: the harm vanishes from the forecast the instant it
is realised.  The term can only produce cooperation indirectly, through the approach step: moving next to
Blue's coin turns "own coin, Red far" into "threat", which is penalised, so a Red that never approaches
never faces the steal decision.  This suggests the two phases: early, the critic knows the coarse fact "my
coin on the board is risky" before the fine one "only when the other agent is adjacent"; read from Blue's
seat the coarse fact rewards removing any Blue coin, i.e. stealing (anti-social phase); once the
adjacency-specific fact is learned the approach penalty dominates and cooperation emerges through
avoidance.  The avoidance part is the perspective-taking effect the project is about (Red applies what it
knows about being threatened to Blue), and it is what the control cannot produce.  All of this is a
hypothesis; the critic probe over training (Section 6) distinguishes the phases: at 70k Blue-colour coins
should score low from Blue's seat regardless of Red's position, by 200k the adjacency term should dominate.
Note that every prospect-based variant (level, scaled, change, counterfactual) shares the "harm vanishes
from the forecast" defect, because all of them score the next board; only outcome-based terms (reward
signal, imagined reward) penalise the steal itself.

### 4.5 Why the gap terms fail

SIA and IA penalise *differences* and are indifferent between equal-good and equal-bad outcomes; in a
symmetric game they have no directional preference (3.1).  With the value signal they inherit the
forecast problem of 4.3 (a forecast gap is near zero in symmetric positions).  With the reward signal the
gap of smoothed rewards is also near zero at the symmetric grabbing equilibrium, so there is no gradient
until asymmetric episodes appear; both IA-reward and SIA-reward were still slowly rising at 2M steps.
Hughes et al. (2018) report that disadvantageous inequity aversion works in their Harvest "via
punishment" with a fining beam, and that advantageous inequity aversion works in Cleanup, a public-goods
game; the Coin Game has neither a punishment action nor a public good, so a weak IA result there does not
contradict their findings and should not be presented as such.  IA-value's success is consistent with its
envy term containing `+alpha * V_i(o_i')`, i.e. the control's mechanism.

## 5. Proposed reformulations

### 5.1 Imagined reward through one's own reward model (`--signal imagined`)

The project's premise is: *I cannot observe your reward, but I assume you are like me.*  The direct
implementation of that premise is a **reward model**, not a value function.  Agent `i` learns
`rhat = f_i(o^t, o^{t+1})` by supervised regression on its own transitions (`f_i` predicts `r_i^t` from
`(o_i^t, o_i^{t+1})`), then imagines the other's reward as `f_i(o_j^t, o_j^{t+1})` and feeds it into the
existing intrinsic-reward machinery (smoothing, EI/SVO/SIA/IA, unchanged).  Properties:

* No reward access, as required.  No forecast and no telescoping (4.3): the imagined quantity is a per-step
  reward, and its sum measures what `i` believes `j` received.
* Reward scale, so `alpha` has the same meaning as in the literature's methods and their `alpha` grids
  apply.
* It reduces to the reward-signal methods exactly when the other agent's reward function equals `i`'s and
  `f_i` is exact.  The gap between `imagined` and `reward` then measures the cost of the assumption
  "the other is like me", which is the question the paper asks.
* Assumption made explicit: rewards are transferable across agents up to a change of perspective.  This
  is the same assumption Bussmann et al. (2019) make ("some types of reward ... may generalize across
  agents") and it fails when agents have different reward functions; that is a limitation to state, and
  heterogeneous-agent experiments would quantify it.

In the Coin Game `f_i` is learnable exactly (+1 when I land on a coin, -2 when the other agent lands on my
coin), so the prediction is that imagined-EI approaches EI-reward (0.996) rather than the value signal's
0.80.  In Harvest the analogous model ("eating an apple gives +1") is equally learnable.

### 5.2 Relative scaling of any social term (`--social-scale`): hygiene, not a fix

Standardise `X` to the advantage's scale within the batch, `Xtilde = alpha * X * std(A) / std(X)` (with a
floor on `std(X)`, since in the PD `X` was numerically zero), so that `alpha = 1` means "the other counts
as much as my own advantage".  What this buys: `alpha` has the same meaning across games, signals and
training phases (today `X` is on the value scale and `A` on the TD-error scale, and both drift); the
method and its control are compared at the same relative weight; and domination of the advantage becomes
a deliberate choice instead of the accident of 4.2.  What it does not do: it changes neither the direction
of the term, nor its one-step myopia (4.1), nor the forecast problem (4.3).  Apply it to every variant,
including the controls, and report `alpha` as a relative weight.  Like advantage normalisation, it makes
the effective weight depend on batch statistics; that is a known trade-off, not a new one.

### 5.3 Change instead of level: not a fix (kept for the record)

An earlier draft proposed `X_i = alpha * [V_i(o_j^{t+1}) - V_i(o_j^t)]` as an advantage coefficient,
claiming it removes the level noise and the frozen-board pathology.  That claim was wrong.  `V_i(o_j^t)`
does not depend on agent `i`'s action at `t` (the observation is fixed before the action), so subtracting
it is subtracting a baseline: the expected policy gradient is *identical* to that of the level term
(Williams 1992), and so is the objective (4.1), pathologies included.  The only differences are second
order: lower variance of the estimate and, because `A + X` is normalised jointly, a smaller `std(X)` that
leaves the advantage more weight, which is exactly what 5.2 does explicitly.  Fed in as a *reward*
through GAE instead, the difference is potential-based and inert (4.3).  So the "change" variant is either
the current method with a baseline or nothing, and its proposed control coincides with the existing
control in expectation.  Dropped as a candidate; the baseline could be kept as a variance-reduction detail
of the current term if that term is reported at all.

### 5.4 Counterfactual influence on the other's prospects (later)

The causal effect of my action on the other's prospects is
`D_i^t = V_i(o_j^{t+1}) - E_{a' ~ pi_i}[ V_i(o_j^{t+1}) | s^t, a' ]`, the realised next value minus what
my other actions would have produced.  Two ways to use it, with very different status:

* As an *advantage coefficient* it is the level term minus a baseline that does not depend on my action
  (the same holds if the baseline also conditions on the other agents' actions, as in COMA-style
  counterfactual baselines), so the expected gradient and the objective are those of the current method;
  only the variance changes.  Same verdict as 5.3.
* As a per-step *reward* fed through GAE it is a different objective: maximise the discounted sum of my
  causal contributions to the other's outlook.  It does not telescope (it is not a difference of one
  function at consecutive states), so it is not inert, and it is far-sighted.  It is also not the other's
  return: it credits me for improving the other's outlook without debiting the later realisation of that
  outlook by the other.  This is in the spirit of the counterfactual influence reward of Jaques et al.
  (2019), who reward the effect of an agent's action on the other agents' *actions*.

However, the reward form still scores the other's *next board*, so it inherits the defect of 4.4: in the
Coin Game, stealing turns Blue's "threat" board into a "safe" board and `D_i` is *positive* for the steal
(the realised -2 is invisible to a forecast).  A counterfactual on prospects therefore rewards realising
threats just as the level term does.  The counterfactual idea is sound only when applied to an
outcome-based quantity, e.g. imagined rewards (5.1): `rhat_j^t - E_{a'}[rhat_j^t | s^t, a']`, which credits
me for the part of Blue's imagined reward that my action caused.  Only worth considering in that form, and
only if 5.1 leaves something unexplained.

## 6. Protocol and pre-registered predictions

Same protocol as before: PPO settings selected per method on tuning seeds (`coin.sh ppo`), fresh seeds for
the reported numbers, the own-value control and plain PPO in every table, the reward-access method as the
ceiling.  Predictions, stated before running:

| experiment | if the hypotheses of 4.4 hold | if they do not |
|---|---|---|
| own-value level `alpha * V_i(o_i')` as a shaping *reward* (through GAE) | cooperation > 0.5 (changes the objective) | no effect would mean the myopic use, not the level, is what matters |
| own-value difference `alpha * [gamma V_i(o_i') - V_i(o_i)]` as a shaping *reward* (through GAE) | cooperation 0.50 (policy-invariant) | cooperation > 0.5 would mean the shaping argument is wrong or PPO's finite-sample dynamics matter more than the limit |
| critic probe on EI-value checkpoints (50k, 200k, 2M), boards scored from the other's seat | at 50-70k: any Blue-colour coin scores low regardless of Red's position (coarse risk); by 200k: "Blue's coin with Red adjacent" scores clearly below "Blue's coin with Red far" (adjacency-specific); "safe" > "threat" throughout | no such ordering, or no change over training: the EI-value story in 4.4 is wrong |
| imagined-EI (5.1) | approaches EI-reward (about 1.0 cooperation) | stays near 0.8 or below: the reward model, not the signal, is the bottleneck |
| current EI-value and control with relative scaling (5.2) at `alpha` in {0.3, 1, 3} | the ranking EI-value > control survives at moderate relative weights | if the gap only exists when `X` dominates, the social increment is a large-weight artefact |

The comparison that decides how the paper is framed is *other-regarding term vs. own-value control at the
same PPO settings*, not *value signal vs. plain PPO*.

## 7. Open questions

* Transfer to Harvest: the "leave the other's coin so I cannot be exploited" structure has no direct
  analogue there, so the control's effect may not carry over while the other-regarding effect might, or
  the reverse.  This is the real test of generality.
* Heterogeneous agents: all methods here assume symmetric reward functions; the imagined-reward
  formulation makes the assumption explicit and testable.
* Non-stationarity: `V_i` is a forecast under a moving joint policy; none of the arguments above rely on
  convergence of the critic, but the early-training behaviour (spikes, EI-value's anti-social phase) does.
* Statistics: 8 seeds separate the groups in Section 3.2 but not every pair inside the value-signal
  cluster; the differences to the control that matter should be re-tested with the reformulated methods
  rather than with more seeds of the current one.

## Appendix: every variant in one notation

`V(o; w_i)` is agent `i`'s critic (parameters `w_i`), `o_j^{t+1}` agent `j`'s next observation, `A_i^t` agent
`i`'s GAE advantage, `F_i(z)` one of the formulations of Section 1 applied to `z_i = (z_i1, ..., z_iN)`.  A
social term enters either as a **coefficient** `X_i^t` (PPO uses `A_i^t + X_i^t`, normalised per minibatch;
judged one step ahead) or as a **reward** `Y_i^t` (PPO uses `r_i^t + Y_i^t`, then GAE; carried through the
return).

```
reward signal (Hughes et al.)   z_ij = e_j,  e_j^t = gamma*lambda*e_j^{t-1} + r_j^t          Y_i = F_i(z)   X_i = 0
value signal (current)          z_ij = V(o_j^{t+1}; w_i)  (incl. j = i)                       X_i = F_i(z) * (1 - done^{t+1})
own-value control (SVO phi=0)   X_i = alpha * V(o_i^{t+1}; w_i)                                never reads o_j
5.1 imagined reward             theta_i = argmin E[(f(o_i^t,o_i^{t+1};theta_i) - r_i^t)^2]   (own transitions)
                                rhat_j = f(o_j^t, o_j^{t+1}; theta_i),  rhat_i = r_i
                                z_ij = ehat_j,  ehat_j^t = gamma*lambda*ehat_j^{t-1} + rhat_j^t   Y_i = F_i(z)   X_i = 0
5.2 relative scaling            Xtilde_i = alpha * X_i * std_B(A_i) / max(std_B(X_i), eps)   (F with unit weight)
5.3 change (not a fix)          X_i = alpha*[V(o_j^{t+1};w_i) - V(o_j^t;w_i)] = current X - alpha*V(o_j^t;w_i);
                                the subtracted term is action-independent -> same expected update as current
5.4 counterfactual (reward)     D_i = V(o_j^{t+1};w_i) - sum_a' pi_i(a'|o_i^t) V(o_j^{t+1}(a');w_i),   Y_i = alpha*D_i
PPO coefficient (all)           c_i = A_i + X_i, normalised (c - mean_B c)/std_B c, with A_i from r_i + Y_i
```

**EI, worked example.**  Two players, Red = `i`, Blue = `j`.  Red stands next to Blue's coin; it can steal
(Red +1, Blue -2, new coin in Red's colour) or leave (Blue takes it next step: Blue +1, new coin in Red's
colour).  Boards scored by Red's critic from Blue's seat: "threat" (Blue's coin, Red adjacent) = 7,
"safe" (Red-colour coin on the board) = 10.  Red's own immediate reward: steal 1, leave 0.

```
variant                 EI term                                          steal          leave      result
reward EI (lambda=0)    Y = alpha * r_blue                               1 - 2 alpha    alpha      leave if alpha > 1/3
value EI (current)      X = alpha * V(o_blue^{t+1}; w_red)               1 + 10 alpha   7 alpha    steal, any alpha
own-value control       X = alpha * V(o_red^{t+1}; w_red)                1 + 7 alpha    10 alpha   leave if alpha > 1/3
5.1 imagined EI         Y = alpha * f(o_blue^t, o_blue^{t+1}; theta_red) 1 - 2 alpha    alpha      leave if alpha > 1/3
5.2 scaled value EI     X = alpha * V(o_blue^{t+1}) * std(A)/std(X)      same ranking as value EI
5.3 change EI           X = alpha * [V(o_blue^{t+1}) - V(o_blue^t)]      1 + 3 alpha    0          steal (same ranking)
5.4 counterfactual EI   Y = alpha * [V(o_blue^{t+1}) - avg_a' V(...)]    1 + 3 alpha    0          steal
```

Every prospect-based row rewards the steal at the moment of decision, because the -2 is not in the next
observation while the respawned coin makes Blue's board "safe"; the outcome-based rows penalise it.  The
value term can produce cooperation only indirectly, by penalising the *approach* step (Section 4.4).

## References

* Bussmann, B., Heinerman, J., Lehman, J. (2019). Towards Empathic Deep Q-Learning. arXiv:1906.10918.
* Devlin, S., Kudenko, D. (2011). Theoretical considerations of potential-based reward shaping for multi-agent systems. AAMAS.
* Devlin, S., Kudenko, D. (2012). Dynamic potential-based reward shaping. AAMAS.
* Fehr, E., Schmidt, K. M. (1999). A theory of fairness, competition, and cooperation. QJE.
* Foerster, J. et al. (2018). Learning with Opponent-Learning Awareness. AAMAS. arXiv:1709.04326.
* Foerster, J. et al. (2018). Counterfactual Multi-Agent Policy Gradients (COMA). AAAI. arXiv:1705.08926.
* Hughes, E. et al. (2018). Inequity aversion improves cooperation in intertemporal social dilemmas. NeurIPS. arXiv:1803.08884.
* Jaques, N. et al. (2019). Social Influence as Intrinsic Motivation for Multi-Agent Deep RL. ICML. arXiv:1810.08647.
* Lerer, A., Peysakhovich, A. (2017). Maintaining cooperation in complex social dilemmas using deep RL. arXiv:1707.01068.
* Littman, M. (1994). Markov games as a framework for multi-agent reinforcement learning. ICML.
* Lu, X., Schwartz, H., Givigi, S. (2011). Policy invariance under reward transformations for general-sum stochastic games. JAIR. arXiv:1401.3907.
* Madhushani, U. et al. (2023). Heterogeneous Social Value Orientation Leads to Meaningful Diversity in Sequential Social Dilemmas. arXiv:2305.00768.
* McKee, K. R. et al. (2020). Social diversity and social preferences in mixed-motive reinforcement learning. AAMAS.
* Ng, A. Y., Harada, D., Russell, S. (1999). Policy invariance under reward transformations: theory and application to reward shaping. ICML.
* Raileanu, R., Denton, E., Szlam, A., Fergus, R. (2018). Modeling Others using Oneself in Multi-Agent RL. ICML. arXiv:1802.09640.
* Schulman, J. et al. (2016). High-dimensional continuous control using generalized advantage estimation. ICLR.
* Schulman, J. et al. (2017). Proximal policy optimization algorithms. arXiv:1707.06347.
* Schwarting, W. et al. (2019). Social behavior for autonomous vehicles. PNAS 116(50).
* Shapley, L. S. (1953). Stochastic games. PNAS.
* Sutton, R. S., McAllester, D., Singh, S., Mansour, Y. (2000). Policy gradient methods for RL with function approximation. NeurIPS.
* Tucker, G. et al. (2018). The Mirage of Action-Dependent Baselines in Reinforcement Learning. ICML. arXiv:1802.10031.
* Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning.
