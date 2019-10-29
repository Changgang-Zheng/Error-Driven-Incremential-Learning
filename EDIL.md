## Error-Driven Incremental Learning in Deep Convolutional Neural Network for Large-Scale Image Classification
> We developed a training algorithm that grows a network not only incrementally but also hierarchically. Classes are grouped according to similarities, and self-organized into levels. The newly added capacities are divided into component models that predict coarse-grained superclasses and those return final prediction within a superclass. Importantly, all models are cloned from existing ones and can be trained in parallel. These models inherit features from existing ones and thus further speed up the learning. Our experiment points out advantages of this approach, and also yields a few important open questions.

<center><img src="/Users/changgang/Documents/Study Notes/Data/Photos/Screen Shot 2019-01-30 at 3.45.50 PM.png"  alt="Transfer Learning" width="900"/></center>

<center><img src="/Users/changgang/Documents/Study Notes/Data/Photos/Screen Shot 2019-01-30 at 3.48.19 PM.png"  alt="Transfer Learning" width="400"/></center>

### Algorithm
* all N0 classes are in one single superclass and predicted by one model L0. 
* When new classes come and the superclass size increases to N1, we have two choices to make the model bigger. 
	* One choice is simply extending L0 to L′0 by inserting more output units. Thus, L0 is a leaf model by itself.
	* The second choice that substantially scales up the capacity is partitioning the superclass into K superclasses and clone L0 into several new leaf models L1, L2, . . . , LK to predict within each of these new superclasses.=== A branch model B with K final output units is also cloned from L0 to direct the prediction to the correct leaf model on a given input sample. 
	
#### Three problems we need to address:
##### 1) How to partition a superclass; 
Using current model L0 on samples drawn from all the data, including both new and old. A validation set of N0 are tested through L0, and calculating a confusion matrix C ∈ RN0×N0 from the output. The entry Cij denotes the probability that the i-th class is predicted to j-th class, which also measures the similarity between class i and j. We then use spectral clustering partition to split N0 classes into K clusters based on the confusion matrix. 

##### 2) How to re-train these models with changed data and objectives; 
Instead of training from scratch, we train these new leaf models incrementally by copying L0’s param- eters as initialization and using random initialization on the remaining new parameters (i.e. the weights connecting the units of the last hidden layer to the newly added output layer units). This training is much more efficient than from scratch.

<center><img src="/Users/changgang/Documents/Study Notes/Data/Photos/Screen Shot 2019-02-09 at 10.26.59 AM.png"  alt="Transfer Learning" width="400"/></center>

We also evolve a new leaf model L′0 to have N1 output units, trained with flat incremental with data of N1 classes. Finally, we clone the branch model B by copying parameters from L′0. As illustrated in Figure 4, and simply sum up the softmax units belonging to each superclass as the predicted probability of that superclass.

<center><img src="/Users/changgang/Documents/Study Notes/Data/Photos/Screen Shot 2019-01-30 at 3.48.35 PM.png"  alt="Transfer Learning" width="400"/></center>

##### 3) Most importantly, how to decide the best strategy out of the two alternatives.
We adopt a simpler strategy to simply let the two compete. This is illustrated in Algorithm 1.

<center><img src="/Users/changgang/Documents/Study Notes/Data/Photos/Screen Shot 2019-01-30 at 3.48.43 PM.png"  alt="Transfer Learning" width="400"/></center>

The process of cloning and incremental leaf model training automatically transfers features learned from the old model L0 to the new leaf and branch models


### Incremental Learning

Incremental learning from a single superclass can be generalized to a deeper hierarchy, at each step receiving new batches of classes (Algorithm 2).

* The first step is to dis- tribute all the new classes into the existing superclasses by error-driven preview in a coarse granularity of superclass (line 1 to 6).
* Next, for each leaf model l, we grow it using flat increment or clone increment by Algorithm 1, discussed previously. 

<center><img src="/Users/changgang/Documents/Study Notes/Data/Photos/Screen Shot 2019-01-30 at 3.48.49 PM.png"  alt="Transfer Learning" width="400"/></center>


