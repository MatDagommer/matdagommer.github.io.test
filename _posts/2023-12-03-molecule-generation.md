---
title: "Data-efficient framework for molecule generation"
date: 2023-12-03
---

Often, we have little data at our disposal in chemical studies. Experiments can be time-consuming, both in terms of preparation and reaction times (some experiments are run for months!). In addition, chemists are limited by the available catalogues from suppliers, and some molecules are only available on-demand, which adds additional time-constraint. On top of that, equipment is needed for controlling temperature and pression conditions during reaction as well as storage space. All this constraints limit the number of molecules that can be tested by chemistry teams. 

In materials and therapeutic discovery, one is often interested in generating molecules with desired properties by using generative models. These models are neural networks trained with large datasets that can infer new chemical compounds starting from the properties of the molecule (e.g., logP, QED, SA). For these models to work properly, large datasets are required for training. In this case, data would correspond to pairs of molecule structure plus their properties, but properties often require experiments to be quantified. 

Since collecting chemical properties data is expensive and time-consuming, datasets often contain ~100 data points. This is too little to train deep neural networks such as variational autoencoders or generative adversarial networks, typically used for generation tasks. Is molecule generation still possible in that case?

I asked the question myself, as I was facing this exact problem. I’ve come across the following paper, Data-efficient Graph Grammar Learning for Molecule Generation, which proposes an original, elegant way of tackling this problem. After spending some time deciphering the paper, I decided to make this post to highlight the big take-aways.


1.	Data-efficient graph grammar learning for molecule generation.

The first term that needs to be demystified is grammar. When we think about grammar, we think about language and how a set of grammatical rules helps us a build a sentence that makes sense by adding words together. Similarly, we can define a grammar for chemistry, which is essentially a set of rules on how to add atoms, or groups of atoms, together.

At a high-level, this is what it could look like:

{Draw rules}

Here, rule 1 corresponds to an initialization rule: we’re adding the first fragment of the molecule we’re building. Rule 2 builds upon rule 1, by adding another functional group. Rule 3 imagines a different combination starting from the same molecule. We can keep on going like this, and list more rules, including several initialization rules.

In practice, rules do not incorporate entire molecules on each side, but rather a set of “non-terminal nodes” connected to “anchor points” on the left side, on top of which a group is attached on the right side. 

Once we have a grammar at our disposal, we can start building! Production rules can be combined sequentially to come up with actual molecules. However, there is no guaranty that the grammar we just created is going to yield molecules with the properties we are looking for! 

To achieve this, we can ask expert chemists to design manually a set of rules that is likely to produce molecules with the right properties. However, this process is tedious and error prone. In general, such sets of rules work at an atom-level (adding one atom at a time), which is too fine-grained when we know some functional groups and molecular substructures are instrumental in providing a molecule with the right properties. 

Let’s forget about the experts (no offense) and try to come up with rules ourselves. We could start from a database of existing molecules and break them down into chunks. Doing so naturally yields production rules: we have two chunks, part A and B. With either A or B on the left side, we get AB on the right side. In the figure, I’m showing what this process looks with aspirin. I break down the molecule by removing bonds 1 to 4 and retrieve the associated production rule.

By processing all molecules from the database, we eventually come up with a list of rules. Yet, we still have no guaranty that the molecules generated with this new grammar will have the right properties! 

This is where machine learning comes into play: we can train a model to learn where to break down molecules from our database to retrieve rules that will likely yield molecules with the right properties when combined. This is what this process looks like:
![image](https://github.com/MatDagommer/skills-github-pages/assets/64140055/8671dfcb-6883-47da-8db4-bd21dcfc4b12)


I am keeping things high-level; the underlying process is slightly more complicated. But essentially, what happens is the following:
	Molecules from the database are converted into graphs.
	The neural network predicts the probability of removing each bond (edge) from the molecule (graph) during the breakdown step. This is the vector p(X) in Fig. XXX.
	Edges to be removed are selected using random sampling based on those probabilities. Each edge is sampled independently from the others. 
	A production rule is derived from the resulting molecule cut and added to the grammar.
	Once all molecules have been processed, the final grammar is used to generate a set of new molecules.
	The fitness of the generated molecules is evaluated and used as feedback to train the neural network.

Step 6 is non-trivial: in fact, some the operations that got us a set of new molecules are non-differentiable.  The first one is the random sampling step. The second one is the way we combine PRs to retrieve new molecules (need precisions!). In these conditions, it is impossible to compute a gradient analytically. 

Let’s look at the typical objective function:
max┬θ⁡〖E_X [∑_i▒λ_i  M_i (X)]〗

Here, M_i’s represent metrics we’re trying to maximize (e.g., logP, QED, SA). We can give more or less weight to metrics relative to the others by adjusting coefficients λ_i. Now let’s compute the gradient of this function with respect to the weights of our neural network, θ:


$\begin{gathered}
\nabla_\theta \mathbb{E}_X\left[\sum_i \lambda_i M_i(X)\right]=\nabla_\theta \int_X d X p(X) \sum_i \lambda_i M_i(X) \\
=\int_X d X \sum_i \lambda_i M_i(X) \nabla_\theta p(X)
\end{gathered}$

Here, we use the following trick: ∇_θ p(X)=p(X) ∇_θ  log⁡p(X):

=∫_X▒〖dXp(X) ∑_i▒λ_i  M_i (X) ∇_θ  log⁡p(X) 〗
=E_X [∑_i▒λ_i  M_i (X) ∇_θ  log⁡p(X) ]

Almost there! The problem we’re facing with this formula is that we cannot compute the exact expected value, for the simple reason that the space of selected edges is too big! For instance, a set of 100 molecules with an average of 5 bonds yields 2^((5 ×100))=2^500 possible combinations. On top of that, the space of molecules that can be generated with each rule is potentially huge. 
 
We use a heuristic approach to estimate this expected value: Monte-Carlo. Starting from distribution p(X), we sample selected edges N times and derive N different sets of generated molecules. We average the result over all samplings:

≈1/N ∑_(n=1)^N▒∑_i▒〖λ_i M_i (X) ∇_θ  log⁡p(X) 〗

There is nothing left we cannot compute in this expression! We can now train the network with gradient descent using this approximation of the gradient. 

To recap: we’ve just trained a neural network that predicts where to cut molecules, so that the generated grammar yields molecules with desired properties when chaining rules together.

I want to emphasize how fundamentally different this approach is from variational autoencoders and generative adversarial neural networks: such models are trained to learn a distribution of the training set in a latent space, and how to decode representations from this latent space back to the molecule space. A decoder model is typically used for the latter task. That means these networks directly return molecules. This is not the case here!

Note that I completely eluded what the actual neural network looks like. This is well-explained in the original paper and does not affect the general understanding of the method.
![image](https://github.com/MatDagommer/skills-github-pages/assets/64140055/091e3a3e-a776-4173-9559-e0e2fd7ec361)


