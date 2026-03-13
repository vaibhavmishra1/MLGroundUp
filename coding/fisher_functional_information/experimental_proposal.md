Research Proposal: A Fisher Fundamental Theorem for Neural Network Training via Functional Information
---

Part I: Five Candidate Ideas
Idea A — Functional Information Monotonicity During Training
Claim: Define functional information of a model at epoch t as:

$$I_f(t) = -\log_2 P_\theta(\mathcal{L}(\theta) \leq \mathcal{L}(\theta_t))$$

where the probability is over random initializations from the prior. Track this quantity epoch-by-epoch and test whether it increases monotonically.

Feasibility: High. Requires sampling random networks and evaluating loss. No gradient computation needed for the measurement itself.
Novelty: Nobody has empirically measured this specific quantity during training. The paper [arXiv:2501.18812] estimates the probability of sampling a trained network, but does not track the trajectory.
Simplicity: The definition directly instantiates Szostak's formula for neural networks.
Idea B — Fisher-Rao Path Length as a Complexity Measure During Training
Claim: Compute the cumulative Fisher-Rao path length along the training trajectory and show it correlates with generalization gap.

Feasibility: Medium. Requires computing the Fisher information matrix (or diagonal approximation) at each step.
Novelty: Low. Liang et al. (AISTATS 2019) already proposed the Fisher-Rao norm as a complexity measure. The path-length variant is incremental.
Simplicity: Medium.
Idea C — Replicator Dynamics Inside Softmax Attention
Claim: Show that attention weight updates in a transformer follow replicator dynamics on the probability simplex, and verify the Shahshahani gradient flow structure.

Feasibility: Medium. Requires careful mathematical analysis of attention gradient updates.
Novelty: Medium. The Neural Replicator Dynamics paper (2019) already connects softmax policy gradient to replicator dynamics, though not specifically for attention.
Simplicity: Low. Transformers introduce many confounding factors.
Idea D — Evolution Strategies vs SGD: Comparing Functional Information Trajectories
Claim: Train the same network on the same task using (a) SGD and (b) natural evolution strategies (NES). Measure functional information trajectories for both and compare their rates.

Feasibility: High. Both algorithms are standard.
Novelty: Medium. Interesting comparison but the result may be predictable (SGD is more sample-efficient).
Simplicity: High. But the insight may be shallow.
Idea E — A Fisher Fundamental Theorem Analog for SGD
Claim: In the replicator equation, Fisher's fundamental theorem states:

$$\frac{d\bar{f}}{dt} = \text{Var}(f)$$

Propose and test the neural network analog: the rate of functional information increase during training is governed by the variance of per-sample loss gradients (a Fisher-information quantity). Specifically:

$$\frac{dI_f}{dt} \approx \alpha \cdot \text{Tr}(F(\theta_t))$$

where F is the Fisher information matrix and alpha is a proportionality constant. This connects the rate of information accumulation to the local curvature of the statistical manifold.

Feasibility: High. Both quantities (functional information rate via finite differences, Fisher trace via gradient variance) are computable.
Novelty: High. No existing work tests this specific relationship.
Simplicity: The experimental setup is straightforward; the insight is deep.
---

Part II: Argument and Selection
Eliminating Ideas B and C
Idea B (Fisher-Rao path length) is incremental over Liang et al. 2019. The novelty is insufficient to justify a paper. Idea C (replicator dynamics in attention) is mathematically interesting but experimentally complex — isolating the replicator structure inside a full transformer involves too many confounds for a clean experiment.

Eliminating Idea D
Idea D (ES vs SGD comparison) is feasible but the scientific yield is unclear. We already know SGD is more sample-efficient. Showing that both increase functional information at different rates confirms a known qualitative fact without revealing new structure.

Choosing Between A and E
Idea A (monotonicity of functional information) is necessary groundwork. Idea E (Fisher fundamental theorem analog) is the deeper result. But E requires A as a prerequisite — you cannot study the rate of change of a quantity without first establishing that the quantity is well-defined and behaves reasonably.

Decision: Combine A and E into a single two-part experiment.

The first part establishes that functional information increases during training (the "zeroth-order" result). The second part tests whether its rate of increase is governed by a Fisher-information quantity (the "first-order" result). Together, this constitutes a Fisher Fundamental Theorem for Neural Network Training.

---

Part III: The Selected Experiment
Title
"Does Training Obey a Fisher Fundamental Theorem? Measuring Functional Information Accumulation in Neural Networks"

Mathematical Setup
Definition 1 (Functional Information of a Model). Let theta_0 ~ pi(theta) be a random initialization from prior pi (e.g., Kaiming normal). Let L(theta) be the loss on a fixed dataset. The functional information at training step t is:

$$I_f(t) = -\log_2 P_{\theta_0 \sim \pi}\bigl[\mathcal{L}(\theta_0) \leq \mathcal{L}(\theta_t)\bigr]$$

This is exactly Szostak's definition: the negative log-probability that a random configuration meets or exceeds the current model's performance.

Definition 2 (Empirical Fisher Trace). The empirical Fisher information trace at step t is:

$$\hat{T}F(t) = \frac{1}{|B|} \sum{(x,y) \in B} \left\| \nabla_\theta \log p_{\theta_t}(y|x) \right\|^2$$

This equals the trace of the empirical Fisher information matrix and is trivially computable as the mean squared gradient norm over a batch.

Hypothesis (Fisher Fundamental Theorem Analog).

The rate of functional information increase is approximately proportional to the Fisher trace:

$$\frac{\Delta I_f}{\Delta t} \approx \alpha(t) \cdot \hat{T}_F(t)$$

where alpha(t) is a slowly varying proportionality function.

Rationale. In population genetics, the rate of mean fitness increase equals the variance of fitness (Fisher's fundamental theorem). The variance of fitness under the replicator equation equals the Fisher speed on the simplex. In neural network training, the Fisher trace measures how "spread out" the per-sample gradients are — the analog of fitness variance. If training on the parameter manifold is structurally parallel to replicator dynamics on the simplex, we should observe the same rate-variance relationship.

Experimental Design
Step 1: Compute I_f(t) at each epoch.

For a set of M = 10000 random initializations theta_0^(1), ..., theta_0^(M):

Evaluate L(theta_0^(m)) for each m (forward pass only, no training).
At each epoch t, compute:
$$\hat{F}(t) = \frac{1}{M} \sum_{m=1}^{M} \mathbf{1}\bigl[\mathcal{L}(\theta_0^{(m)}) \leq \mathcal{L}(\theta_t)\bigr]$$

$$\hat{I}_f(t) = -\log_2 \hat{F}(t)$$

Note: The random network losses need only be computed once and cached.

Step 2: Compute Fisher trace at each epoch.

At each epoch t, compute:

$$\hat{T}F(t) = \frac{1}{|B|} \sum{(x,y) \in B} \left\| \nabla_\theta \ell(\theta_t; x, y) \right\|^2$$

This is a single backward pass over a batch.

Step 3: Test the relationship.

Plot I_f(t) vs epoch. Test for monotonicity (Part A of the result).
Plot dI_f/dt (finite difference) vs T_F(t). Test for proportionality (Part E of the result).
Fit a linear or log-linear model: dI_f/dt = alpha * T_F(t) + epsilon. Report R-squared.
Models and Datasets (Small Scale, High Rigor)
| Setting | Model | Dataset | Parameters |
|---------|-------|---------|------------|

Setting 1: MLP (2 hidden layers, 256 units) on MNIST (~200K params)
Setting 2: Small CNN (3 conv + 2 FC) on CIFAR-10 (~500K params)
Setting 3: Small Transformer (2 layers, 4 heads) on a text classification task (~1M params)
These are deliberately small to allow M = 10000 random samples and full Fisher trace computation without approximation.

Controls and Ablations
Learning rate sweep: Test whether the relationship alpha(t) changes with learning rate.
Optimizer comparison: SGD vs Adam vs SGD+momentum. The hypothesis predicts the relationship holds across optimizers (since it is a geometric property, not an optimizer property).
With and without regularization: L2 regularization changes the loss landscape. Test whether the theorem still holds.
Random labels control: Train on random labels (no learnable structure). Functional information should still increase (the model memorizes), but the Fisher trace relationship may break down, revealing that the theorem requires genuine structure in the data.
What Would Constitute a Positive Result
Part A (Monotonicity): I_f(t) increases monotonically (or near-monotonically with small fluctuations due to estimation noise) across all settings.
Part E (Rate law): dI_f/dt and T_F(t) are strongly correlated (R^2 > 0.8) across training, with a proportionality constant that is approximately constant or slowly varying.
Bonus: The proportionality constant alpha differs meaningfully between true-label and random-label training, suggesting it encodes something about the data-model relationship.
What Would Constitute a Negative Result
I_f(t) is non-monotonic in interesting ways (e.g., decreases during certain training phases). This would itself be a publishable finding — it would disprove the "arrow of learning" hypothesis.
dI_f/dt and T_F(t) are uncorrelated, meaning the evolutionary analogy breaks at the rate level. This would sharply delineate where the analogy holds and where it fails.
Why This Is Novel
No prior work has defined and tracked Szostak functional information during neural network training trajectories.
No prior work has tested a Fisher fundamental theorem analog for SGD.
The paper [arXiv:2501.18812] estimates the probability of sampling a trained network but does not study the trajectory or connect it to Fisher information dynamics.
The replicator-dynamics/natural-gradient equivalence is known mathematically, but the empirical consequence (a rate law connecting functional information increase to Fisher trace) has never been measured.
Why This Is Simple
The experiment requires only: (a) training small networks, (b) evaluating forward passes on random initializations, (c) computing gradient norms. No custom optimizers, no new architectures, no approximations to the Fisher matrix.
Total compute: training one small network + 10000 forward passes (cached once) + gradient norm computation per epoch. This runs on a single GPU in hours.
Why This Is Beautiful
It would establish a quantitative bridge between evolutionary theory and deep learning — not as a metaphor, but as a measurable rate law. Fisher's fundamental theorem of natural selection (1930) is one of the most celebrated results in population genetics. Showing that neural network training obeys an analogous law would:

Give operational meaning to "functional information" in the ML context.
Provide a new theoretical lens for understanding training dynamics.
Suggest that the convergence of SGD is governed by the same mathematical structure as biological adaptation.
---

Part IV: Rigorous Mathematical Grounding
The Evolutionary Side
In the continuous-time replicator equation on the n-simplex Delta_n with the Shahshahani metric g_ij = delta_ij / x_i:

$$\dot{x}_i = x_i(f_i - \bar{f})$$

The mean fitness phi = sum_i x_i f_i satisfies:

$$\frac{d\phi}{dt} = \sum_i x_i (f_i - \bar{f})^2 = \text{Var}_x(f)$$

The Fisher information speed (squared norm of the velocity in Shahshahani metric) is:

$$\|\dot{x}\|_g^2 = \sum_i \frac{\dot{x}_i^2}{x_i} = \sum_i x_i(f_i - \bar{f})^2 = \text{Var}_x(f)$$

So the Fisher speed equals the fitness variance equals the rate of mean fitness increase. This is the information-geometric form of Fisher's theorem.

The Neural Network Side
In SGD on parameters theta in R^d with loss L(theta) = E_{(x,y)}[l(theta; x, y)]:

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)$$

The empirical Fisher information matrix is:

$$F(\theta) = \mathbb{E}_{(x,y)}\bigl[\nabla_\theta \log p_\theta(y|x) \, \nabla_\theta \log p_\theta(y|x)^T\bigr]$$

Its trace is:

$$\text{Tr}(F(\theta)) = \mathbb{E}_{(x,y)}\bigl[\|\nabla_\theta \log p_\theta(y|x)\|^2\bigr]$$

This is the variance of the score function — the direct analog of Var_x(f) in the replicator equation.

The proposed experiment tests whether:

$$\frac{dI_f(\theta_t)}{dt} \propto \text{Tr}(F(\theta_t))$$

holds empirically, which would be the neural network analog of:

$$\frac{d\bar{f}}{dt} = \text{Var}_x(f)$$

The Gap This Fills
The mathematical equivalence between replicator dynamics and natural gradient descent is proven for continuous-time, infinite-population, potential-game settings. Whether the rate law consequence (Fisher's theorem) transfers to the discrete-time, finite-sample, non-convex setting of neural network training is an open empirical question. This experiment directly answers it.

---

Summary
What: Measure Szostak functional information during neural network training and test whether its rate of increase follows a Fisher fundamental theorem analog.
How: Sample 10K random networks, cache their losses, track what fraction achieves the trained model's current loss at each epoch, and correlate the rate of change with the Fisher trace (mean squared gradient norm).
Where: Small MLPs, CNNs, and transformers on standard benchmarks.
Why novel: First empirical measurement of functional information trajectories during training; first test of a Fisher rate law for SGD.
Why simple: Only requires forward passes, backward passes, and counting. No custom methods.
Why beautiful: Would establish a quantitative evolutionary law for learning, bridging Fisher (1930) and modern deep learning.