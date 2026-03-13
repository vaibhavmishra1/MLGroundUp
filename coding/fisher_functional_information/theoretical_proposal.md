Information Geometry and Functional Information in Evolution and AI
Statistical models form curved manifolds: In information geometry, a family of probability distributions is viewed as a Riemannian manifold whose metric is given by the Fisher–Rao information. Amari (2021) emphasizes that the Fisher information matrix “plays the role of the metric tensor” on this statistical manifold, a structure that underlies many applications in AI, physics, and beyond. Modern deep learning can be interpreted in this framework: training a neural network is like moving along a high-dimensional manifold of model parameters, with the Fisher matrix capturing the local curvature of the loss surface. In fact, one can show that natural gradient descent (which rescales updates by the inverse Fisher matrix) exactly performs steepest-descent on this curved manifold. In short, both classical learning and evolutionary dynamics unfold as gradient flows on information-geometric manifolds.

Key point: Probability models are Riemannian manifolds under the Fisher metric. Neural network training traverses this manifold, and taking natural gradients corresponds to moving along Fisher geodesics.
Functional information and selection: Szostak (2003) introduced functional information to quantify how “rare” functional biopolymer sequences are. He defines functional information as [ I(E_x) = -\log_2\bigl[P(\text{random sequence has function}\ge E_x)\bigr], ] the negative log-probability that a random sequence meets or exceeds a given function threshold. Equivalently, if only a fraction (F(E_x)) of all configurations achieves function (E_x), then (I = -\log_2[F(E_x)]). Crucially, this is a global, ensemble-level measure: “functional information is not a property of any one molecule, but of the ensemble of all possible sequences, ranked by activity”. In biological evolution or chemical self-organization, functional information tracks how improbable useful configurations are.

Wong et al. (PNAS 2023) build on this by proposing a “law of increasing functional information.” They argue that in any evolving system where many configurations are tested and selected for one or more functions, the system’s functional information must increase over time. In their words:

“We propose a ‘law of increasing functional information’: The functional information of a system will increase (i.e., the system will evolve) if many different configurations of the system undergo selection for one or more functions.”

This captures an “arrow of complexity” – akin to the second law’s arrow of entropy – driven by selection. As Hazen and Wong note, functional selection is “a very orderly process that leads to ordered states,” distinct from but not violating thermodynamics. They even suggest that information may be as fundamental as mass or energy in physics: “information itself might be a vital parameter of the cosmos, similar to mass, charge, and energy”.

Key point: Functional information measures the rarity of task-performing configurations. When many variants are selected for function, functional information increases over time, suggesting a fundamental selection-driven trend.
Evolutionary dynamics as information flows: Evolutionary selection can be formulated in information-geometric terms. In the replicator equation, the frequency (x_i) of type (i) changes as ( \dot x_i = x_i(f_i - \bar f)) (fitness-driven growth). Information-geometrically, this is a natural gradient flow on the probability simplex with the Fisher–Shahshahani metric. In fact, one finds that Fisher’s fundamental theorem (rate of fitness increase equals variance in fitness) is built into this geometry. Stated succinctly:

“The induced gradient flow is the replicator equation… natural selection forms a gradient with respect to an informatic measure, and hence locally has the direction of maximal information increase. The rate of change of the mean fitness… is given by the informatic variance.”

Thus replicator dynamics push a population uphill in an “informatic” landscape defined by Fisher information. This mirrors how natural gradient ascent in machine learning climbs the likelihood landscape. In both cases the system follows paths of steepest ascent on an information manifold.

Key point: The replicator equation is a natural-gradient ascent on the Fisher-information manifold. Analogously, neural network learning (via natural gradient) moves in the steepest-ascent direction determined by Fisher curvature.
Connecting to large language models (LLMs): These information-geometric and selection principles suggest a deep analogy between evolutionary systems and AI training. Training an LLM via gradient descent (or its stochastic versions) can be seen as an optimization-driven selection among model configurations. In each update, the model “selects” parameter changes that improve its function (e.g. predictive accuracy). Viewed on the information manifold, these updates steadily move the model toward higher likelihood (lower loss), akin to moving toward higher “fitness.”

In this analogy, a trained LLM has acquired higher functional information with respect to its task: its parameters now encode rarer, more specialized configurations that achieve desired outputs. Just as evolution drives up functional information by preferentially retaining functional variants, training drives the model toward parameter regions of high performance (low loss). Di Sipio et al. (2025) note that the training process is constrained by the geometry of Fisher information, which shapes all paths on the loss landscape. One might say that as an LLM learns from diverse data (analogous to many “configurations”), its informational capacity grows.

In concrete terms, both processes satisfy similar mathematical form: replicator dynamics on the probability-simplex and gradient descent on parameter-space are both natural gradient flows. Evolution maximizes mean fitness with respect to Fisher distance; LLM training (especially with natural gradient or second-order methods) maximizes log-likelihood under the same metric. This suggests that LLMs and evolving populations share an underlying “information law”: they accumulate functional/knowledge information through a sequence of selection-like steps.

Key point: Training an AI model can be seen as an information-geometric selection process. Gradient descent on model parameters (in Fisher metric) is analogous to the replicator flow on genotype frequencies.
Novel perspective – an “arrow of learning”: Taken together, these insights hint at a new conceptual synthesis. Just as thermodynamics posits an entropy-increase law and Wong–Hazen propose a complexity/information-increase law for evolution, one can imagine an “arrow of learning” for AI: as models are trained (exposed to many data “configurations” and optimized), their functional information increases. In other words, each training iteration acts like a selective step that raises the model’s information content about the task. This dovetails with neural scaling laws (where more data/compute yield systematically better performance) and suggests those limits reflect information-geometric constraints.

In summary, a unified picture emerges: information is dynamic and law-like in both natural and artificial systems. Information geometry provides the common language – manifolds and metrics – while functional information quantifies the effects of selection or optimization. The parallel implies that principles from one domain (evolutionary biology) may inspire new training algorithms or theoretical limits in AI. For example, viewing LLM training as a Fisher-gradient flow suggests using natural-gradient or evolutionary-inspired updates to efficiently climb the loss landscape. This interdisciplinary viewpoint (supported by Amari’s framework and Wong’s “law”) points toward innovative approaches: treating information not as static data, but as a conserved/growing quantity governed by geometric and selection laws in learning systems.

Summary of connections:
Fisher information endows model parameter space and population simplex with Riemannian geometry.
Functional information (Szostak) measures configuration rarity, and selection tends to increase it.
Replicator dynamics and natural-gradient descent are two faces of the same information-geometric principle.
Hence, AI training and biological evolution can be unified as processes that accumulate information about function, suggesting a generalized “law of information” applying across physics and learning.
Sources: The above synthesis draws on Amari (2021) on information geometry, Szostak (2003) and Hazen et al. on functional information, and recent work applying Fisher geometry to LLM training and evolutionary dynamics. These works together support the idea of information as a dynamic, physically meaningful quantity that guides both natural evolution and AI development.

The synthesis you presented combines several established concepts from information geometry, evolutionary theory, and machine learning optimization. The core ideas are largely consistent with current theoretical work, though the final “law-like” unification is still conceptual rather than formally established. Below is a structured evaluation and clarification of the main components.

⸻

1. Information Geometry of Statistical Models

In information geometry, a parametric family of probability distributions p(x|\theta) forms a Riemannian manifold with metric:

g_{ij}(\theta) = \mathbb{E}\left[\frac{\partial \log p}{\partial \theta_i}\frac{\partial \log p}{\partial \theta_j}\right]

This is the Fisher information matrix.

Properties:
	•	Defines a local notion of distance between distributions.
	•	Determines curvature of the statistical manifold.
	•	Provides a coordinate-invariant metric for optimization.

In machine learning:
	•	Model parameters \theta define a point on the manifold.
	•	Training moves along this manifold to maximize likelihood.

The natural gradient update:

\theta_{t+1} = \theta_t - \eta F^{-1}(\theta_t)\nabla_\theta L(\theta_t)

rescales gradients using the Fisher metric, producing the steepest descent direction in information space, not Euclidean space.

Consequences:
	•	Better alignment with model sensitivity.
	•	Invariance to parameterization.
	•	Often faster convergence.

This framework, largely developed by Shun-ichi Amari, is widely used in theoretical analyses of deep learning.

⸻

2. Functional Information

Functional information (FI) introduced by Jack Szostak is defined as

I(E_x) = -\log_2 F(E_x)

where

F(E_x) = P(\text{random configuration has performance} \ge E_x)

Interpretation:
	•	Measures how rare functional configurations are in the configuration space.
	•	High FI implies functional solutions occupy a small fraction of possible states.

Important properties:
	1.	Ensemble property
FI describes the space of possibilities, not a single sequence.
	2.	Threshold dependent
FI depends on the chosen functional threshold E_x.
	3.	Selection increases FI
Evolution preferentially samples and retains rare functional regions.

In molecular evolution and origin-of-life research, FI quantifies the informational difficulty of discovering functional biomolecules.

⸻

3. Evolutionary Dynamics and Information Geometry

Population dynamics under selection follow the replicator equation:

\dot{x}_i = x_i(f_i - \bar{f})

where
	•	x_i = frequency of type i
	•	f_i = fitness
	•	\bar{f} = mean fitness

The simplex of population frequencies forms a manifold equipped with the Fisher–Shahshahani metric:

g_{ij} = \frac{\delta_{ij}}{x_i}

Under this geometry:
	•	Replicator dynamics become a natural gradient flow.
	•	Mean fitness acts as the potential function.

This leads to a geometric interpretation of Fisher’s Fundamental Theorem:

\frac{d\bar f}{dt} = \mathrm{Var}(f)

meaning selection moves populations along directions of maximal fitness increase under the Fisher metric.

⸻

4. Parallel With Machine Learning Optimization

The analogy with deep learning is mathematically grounded:

Evolution	Machine Learning
genotype frequencies x_i	parameters \theta
fitness f_i	log-likelihood / reward
replicator equation	gradient descent
Fisher–Shahshahani metric	Fisher information matrix
natural selection	optimization

Both systems perform information-geometric gradient flows.

Key similarity:

Both processes iteratively move toward regions of higher functional performance in very large spaces.

⸻

5. Functional Information Increase

Recent work (Hazen, Wong and collaborators) proposes a general principle:

If
	•	many configurations are sampled
	•	selection favors functional performance

then the functional information of the system tends to increase.

Important clarification:

This is not a thermodynamic law but a statistical tendency for evolving systems with:
	•	variation
	•	selection
	•	persistence of successful configurations.

The mechanism is simply biased sampling of configuration space.

⸻

6. Implications for AI Systems

Training large models can be interpreted as:
	1.	Exploring parameter space
	2.	Selecting updates improving function
	3.	Accumulating information about the task

In this interpretation:
	•	parameters encode structured representations of data
	•	optimization gradually restricts parameters to a smaller region of high performance

This resembles an increase in functional information relative to the task distribution.

However, a precise formal mapping between:
	•	Szostak functional information
	•	model parameter entropy

has not yet been fully developed.

⸻

7. Limits of the Analogy

Several differences remain important.

Evolution
	•	decentralized search
	•	stochastic mutation
	•	parallel population sampling
	•	fitness defined by environment

Deep learning
	•	centralized optimization
	•	gradient-based search
	•	deterministic objective functions
	•	human-designed architectures

Thus the analogy is structural rather than literal.

⸻

8. Possible Research Directions

The framework you outline suggests several promising research areas.

Information-geometric training methods
	•	scalable natural gradient approximations
	•	Fisher-Rao regularization
	•	curvature-aware optimizers

Evolutionary learning hybrids
	•	population-based training
	•	evolutionary strategies on parameter manifolds
	•	selection-based neural architecture search

Functional information metrics for AI

Potential definitions:

I_{task} = -\log P(\text{random model achieves performance } \ge E)

This could quantify the difficulty of tasks or rarity of effective models.

Learning thermodynamics

Studying whether training obeys measurable quantities such as:
	•	information accumulation rates
	•	geometric path lengths
	•	energy–information tradeoffs.

⸻

9. Conceptual “Arrow of Learning”

Your proposed idea can be summarized as:

Learning systems tend to move toward regions of configuration space with higher functional information.

This resembles:

Domain	Direction
thermodynamics	entropy increases
evolution	functional information increases
learning	task-relevant information increases

Whether this becomes a formal law will depend on whether measurable invariants or bounds can be derived.

⸻

10. Unified View

Across evolution and machine learning:
	1.	Systems explore large configuration spaces.
	2.	Selection or optimization biases sampling.
	3.	Fisher geometry determines efficient movement.
	4.	Functional structures become increasingly concentrated.

Thus both processes can be viewed as information accumulation under geometric constraints.

⸻

If useful, the next step would be exploring a mathematical unification of:
	•	Fisher information geometry
	•	functional information
	•	scaling laws in large neural networks

which could lead to a general theory of information accumulation in adaptive systems.

Below is a conceptual hypothesis for a new learning algorithm inspired directly by the principles you described: information geometry + functional information + evolutionary selection. The goal is to keep the algorithm as simple as feed-forward training, but introduce a different update rule that combines:
	•	Fisher geometry
	•	functional rarity
	•	selection pressure

This produces a new class of training dynamics.

⸻

Functional Information Gradient (FIG) Learning

Core Idea

Standard neural training optimizes loss.

Instead, the algorithm optimizes functional information increase.

Rather than minimizing:

L(\theta)

the system maximizes:

I_f(\theta) = -\log P(\text{random parameter perturbation performs ≥ current})

This measures how rare the model’s functional behavior is relative to nearby models.

Training therefore pushes the model toward rarer functional regions of parameter space.

⸻

Intuition

Standard gradient descent asks:

Which direction decreases loss fastest?

Functional Information Gradient asks:

Which direction makes the model’s behavior most unlikely to occur randomly?

This introduces selection pressure, not just optimization.

⸻

Algorithm Overview

The algorithm works with three simple steps per batch.

Step 1 — Sample Local Variants

Instead of computing gradients immediately, generate small perturbations:

\theta_i = \theta + \epsilon_i

Where

\epsilon_i \sim N(0,\sigma^2)

Example: 8–16 perturbations.

Each perturbation represents a neighbor model.

⸻

Step 2 — Evaluate Functional Score

Compute task performance:

f_i = performance(\theta_i)

Examples:
	•	classification accuracy
	•	reward
	•	likelihood
	•	task score

Then compute rarity estimate:

F = \frac{\#\{f_i ≥ f_\theta\}}{N}

Functional information estimate:

I = -\log_2(F)

⸻

Step 3 — Information Gradient Direction

Instead of gradients, compute selection-weighted direction:

\Delta \theta =
\sum_i w_i (\theta_i - \theta)

Where

w_i = \exp(\beta (f_i - f_\theta))

This pushes parameters toward rare high-function perturbations.

Update:

\theta_{t+1} = \theta_t + \eta \Delta \theta

⸻

Why This Is Different

Standard methods:

Method	Principle
SGD	local slope
Adam	adaptive slope
Natural Gradient	Fisher curvature
Evolution strategies	random search

FIG learning introduces:

rarity-based selection pressure

The system moves toward configurations that are statistically rare but functional.

⸻

Geometric Interpretation

The algorithm moves on the Fisher manifold but with a new objective:

\max_\theta I_f(\theta)

Meaning the model seeks regions of parameter space where functional behavior occupies minimal volume.

This corresponds to high functional information density.

⸻

Biological Analogy

The algorithm mirrors evolution:

Evolution	FIG Learning
mutations	parameter perturbations
selection	performance filter
population	perturbation set
adaptation	parameter update

But unlike genetic algorithms, this remains continuous and differentiable-like.

⸻

Computational Complexity

Very close to feed-forward training.

If batch forward pass cost is:

C

Then cost per step:

C \times (N+1)

Where N is perturbations (usually small like 8).

Parallelizable on GPUs.

⸻

Optional Fisher Geometry Extension

Perturbations can be sampled using Fisher covariance:

\epsilon \sim N(0, F^{-1})

This aligns exploration with information curvature.

Meaning the algorithm automatically explores meaningful parameter directions.

⸻

Pseudocode

initialize θ

for each batch:

    f_base = evaluate(model(θ))

    for i in 1..N:
        ε_i ~ Normal(0, σ)
        θ_i = θ + ε_i
        f_i = evaluate(model(θ_i))

    weights = softmax(β*(f_i - f_base))

    Δθ = Σ weights * ε_i

    θ = θ + η * Δθ


⸻

Why This Could Be Powerful

This method may:

1. Avoid sharp minima

Because it prefers rare robust solutions.

⸻

2. Encourage modular structure

Rare functional regions often correspond to structured representations.

⸻

3. Work without gradients

Useful for:
	•	non-differentiable objectives
	•	reinforcement learning
	•	program synthesis

⸻

4. Potentially scale well

Because it resembles evolution strategies, which already scale to large models.

⸻

Possible Theoretical Result

One could hypothesize:

\frac{dI_f}{dt} ≥ 0

Meaning training increases functional information over time.

This would be the learning analogue of evolutionary information increase.

⸻

Potential Name

Functional Information Gradient Learning (FIG)

Alternative names:
	•	Selection Gradient Networks
	•	Information Selection Learning
	•	Rarity Optimization

⸻

Minimal Version (Almost SGD)

If you simplify it further:

1 perturbation per step.

Update:

\theta_{t+1} =
\theta_t +
\eta \cdot sign(f_{perturbed} - f_{base}) \cdot \epsilon

This becomes an extremely simple selection-based gradient estimator.

⸻

Why This Might Be Novel

Most ML optimizers minimize loss.

This algorithm explicitly optimizes:

rarity of functional configurations.

That concept comes directly from Szostak functional information and is largely unexplored in neural training.

⸻

If you want, I can also show something more ambitious:

a completely new neural architecture whose neurons implement replicator dynamics instead of weighted sums — which would be a radically new type of neural network inspired by evolutionary information geometry.

does this conversation hold any thing meaningul ?