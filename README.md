# Crosscoders Model Diffing

## Planned structure:
```
crosscodiffs/
├── model_zoo/
│   ├── __init__.py
│   ├── toy_model1.py
│   ├── toy_model2.py
│   ├── crosscoder.py
│   └── ...
├── training/
│   ├── __init__.py
│   ├── train_model.py         # general training loop?
│   ├── train_crosscoder.py
│   └── ...
├── utils/
│   ├── __init__.py
│   ├── data_io.py             # reading/writing data
│   ├── metrics.py             # reconstruction error, correlation analysis
│   └── ...
├── logs/
├── data/
├── experiments/
│   ├── seed_diffs/
│   │   ├── outputs/
│   │   │   └── ...
│   │   ├── models/
│   │   │   └── ...            # Saved model weights
│   │   ├── training/
│   │   ├── analysis/
│   │   └── seed_diffs.ipynb
│   ├── experiment2/
│   │   ├── outputs/
│   │   │   └── ...
│   │   ├── models/
│   │   │   └── ...
│   │   ├── training/
│   │   ├── analysis/
│   │   └── exp2.ipynb
│   └── ...
└── README.md
```

## Ideas
- Identical Architecture, Identical Data, Different Random Seeds
- Identical Architecture, Slightly Different Data Distributions
- Identical Data, Different Architecture
- Identical Architecture, Identical Data, Different Training Objectives
- Models with Differing Numbers of Latent Dimensions
- Additional Variation + Scale-Up, Possibly Multi-Layer or Larger Models
- Chain-of-Thought / Faithful vs. Unfaithful Reasoning Steps
- ...

## Further Collection
### Identical Architecture, Identical Data, Different Random Seeds

see if two “nominally identical” models (same architecture, same hyperparameters, same data) learn latent representations that are effectively the same or if they differ (due to random initialization or nondeterminism)  

Hypothesis:  
- If the data and architecture are the same, two different random initializations lead to rotations / permutations of the same feature set.  
- A crosscoder might learn a near-perfect transformation (it might effectively learn to “un-rotate” or “permute” the latent space).
-low reconstruction error.

Plan:  
1. Train two toy models (call them Model 1.1 and Model 1.2) with the same architecture on the same data, only different random seeds.  
2. Extract their latent representations on a validation dataset.  
3. Train a crosscoder from Model 1.1’s latent to Model 1.2’s latent.  
4. Check reconstruction error, correlation structure.  

Interpretability:  
- If the crosscoder has near-zero MSE or near-perfect correlation, it suggests the two models indeed learned “the same” features up to a transformation.  
- can also do a dimension-to-dimension correlation matrix analysis to see if the crosscoder discovered a near one-to-one mapping or if the features are “mixed.”

### Identical Architecture, Slightly Different Data Distributions

see how changes in the input data distribution affect the learned latent space.

Hypothesis:
- If the data differences are small (e.g., a subset of classes, slight shift in input distribution), we expect the crosscoder to still do well, but possibly see a slight increase in reconstruction error compared to Experiment Series 1.
- If the data difference is large, the crosscoder might fail to find a neat transformation.

same architecture as in Experiment Series 1  
two different training sets, e.g.,  
- Data set A: 0–4 MNIST digits  
- Data set B: 5–9 MNIST digits  
(Or any synthetic dataset that has meaningful differences.)  

Model 2.1 on data A, and Model 2.2 on data B  
Crosscode from latent of 2.1 to 2.2  
reconstruction error, correlation  

possibly:
intermediate data distribution that overlaps partially with each to see if partial overlap fosters partial success in crosscoding

??:  
if certain latent features “disappear” or “emerge” in one model but not in the other- The crosscoder might effectively attempt to “invent” features that Model 2 has but Model 1 does not

can measure how success/failure correlates with how different the data sets are.

### Identical Data, Different Architecture

Now the difference is purely in the model design (e.g., a feedforward vs. a convolutional net, or a 1-layer vs. 2-layer MLP).  

Hypothesis:  
- With the same data, do these architectures learn similar or drastically different latent features?  
- If the architectural difference is small (e.g., same MLP but different hidden dimension), the crosscoder might do moderately well. If it’s large (e.g., MLP vs. CNN with different representation biases), performance might drop.  

Both models see the same training data  
Model 3.1: 1-layer MLP. Model 3.2: 2-layer MLP (or a CNN)  
Training a crosscoder, measureing reconstruction error & correlation  
Possibly varying how big the architecture difference is  

??:  
see if a crosscoder can decode CNN-based features from MLP-based latents, or if the features are fundamentally quite different

### Identical Architecture, Identical Data, Different Training Objectives

Do two models that see the same data but optimize different objectives (or have different regularization or different “heads” like classification vs. autoencoding) learn latent spaces that are mappable?  

Model 4.1: Minimizes standard cross-entropy for classification  
Model 4.2: Minimizes a reconstruction loss (autoencoder style)  
Possibly the same “backbone” but different tasks  

Hypothesis:
- The crosscoder might show that the classification model’s latents are strongly aligned with class boundaries, whereas the autoencoder’s latents might represent pixel-based features  
- We might expect a bigger mismatch and hence higher crosscoder error  

Model 4.1 for classification  
Model 4.2 as an autoencoder
crosscoding between them - can map how well?

??:  
measuring the dimension of the subspace that the crosscoder can easily map onto
might reveal that some “generic features” (like shape or edges) do align, but other tasks-specific features do not  

### Varying Latent Dimensionality

effect of having a different number of latent features. E.g., Model 5.1 has a 16-dim latent, Model 5.2 has a 32-dim latent 

Hypothesis:
- The crosscoder from 16-dim → 32-dim might learn to “spread out” features or “invent” 16 new dimensions. We’d see if some dimensions remain empty or effectively just noise if the second model truly needs more dimensions for the same data  

data the same.  
Fixing the architecture except for the dimension of the last hidden layer  
Training the crosscoder.  
performance. Possibly looking at which dimensions in the 32-dim are used  

??:  
reveals how “redundant” or “compressible” the feature space is between the two models. - may find that if 32-dim is an over-parameterized latent for the same data, it might still be easy to crosscode from 16-dim  
Alternatively, if the 32-dim model truly learns finer distinctions, the crosscoder might have a harder time capturing them  

### Additional Variation + Scale-Up

beyond single-layers or tiny data sets. Possibly using bigger toy tasks or more realistic tasks (e.g., a subset of CIFAR, or a simple synthetic language data, etc.).

more layers.  
attention-based layers.  
different training schedules or optimizers.  
Evaluating generalization of the crosscoder (train crosscoder on a subset of data, test on a new subset).  

Hypotheses:
- As models get deeper and more complex, the crosscoder’s job might get more complicated
- might discover that certain deeper models share similar intermediate “shapes” of features (like edges or corners in CNNs)

### ? Chain-of-Thought / Faithful vs. Unfaithful Reasoning Steps ?

a model may produce internal text tokens or hidden states that represent step-by-step reasoning. We can treat these steps as “latent states” across a sequence.  

see if a crosscoder can map from one model’s chain-of-thought latent to another model’s chain-of-thought latent. If one model’s chain-of-thought is “faithful” (truly capturing the intermediate reasoning steps) but the other is “unfaithful” (the steps are more superficial or do not reflect the final answer’s actual reasoning path), do we see a difference in crosscoder performance?  

- small language-based toy dataset (like arithmetic or symbolic reasoning tasks)  
- Model A: Trained with forced chain-of-thought supervision or some form of intermediate supervision.
- Model B: Trained only on final answers (no forced intermediate steps).
- how well the crosscoder can map A’s intermediate hidden states to B’s (and vice versa).

Hypothesis:
- If B does not truly encode intermediate steps in a faithful manner, the crosscoder might fail to reconstruct B’s internal states from A’s  
- Alternatively, if B’s chain-of-thought–like states are still correlated with A’s, then crosscoder performance may be relatively good

??:  
Checking dimension-by-dimension alignment could reveal whether certain “tokens” or “features” representing partial computations are universal across both models or not  
This could help “prove” or at least provide strong evidence that a model’s chain-of-thought is either real (reconstructable from a genuinely faithful model) or is more “made up”  


## Evaluation:

Reconstruction Error: MSE or L1 between crosscoder output and Model B’s latent  

Correlation / Cosine Similarity: dimension-wise or full-vector correlation  

Possibly cluster or visualize latents (e.g., t-SNE, PCA) to see how the crosscoder is aligning them  

Interpretation:

If the crosscoder fails (high error), that suggests the two latent spaces are significantly different  

If it succeeds (low error), it implies the spaces are isomorphic or nearly so  

## sub-experiments that gradually increase the difference

e.g. Series 2 (Data Variation):
- 2A: Minor difference (train on digits 0–9 vs. digits 0–8).
- 2B: Medium difference (0–4 vs. 5–9).
- 2C: Large difference (MNIST vs. FashionMNIST).

-> stepwise approach, track how crosscoder error evolves as the difference grows

