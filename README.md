# BayesReef

<div align="center">
    <img width=600 src="https://writelatex.s3.amazonaws.com/skzysbnqjpdk/uploads/28047/26883067/1.png?X-Amz-Expires=14400&X-Amz-Date=20180719T010027Z&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJF667VKUK4OW3LCA/20180719/us-east-1/s3/aws4_request&X-Amz-SignedHeaders=host&X-Amz-Signature=5dffaa250b29b7a623380bbc8a6d5b0204f3f1588e71cc9499915e4073a810a9" alt="Flowchart of BayesReef by Jodie Pall, CTDS, 2018" title="Flowchart of BayesReef process. A fusion of multiple sources of data is used to create prior probability distributions on free parameters. Following this, BayesReef is initiated with a vector of free parameters are drawn from the prior. The MCMC sampler uses a Metropolis-Hastings (M-H) algorithm as a basis to accept or reject proposed samples. The sampler terminates when all the allocated samples have been assessed."</img>
</div>

BayesReef is a framework for implementing Bayesian inference on the deterministic model, _pyReef-Core_. It can simulate data from the _pyReef-Core_ stratigraphic forward model (SFM) to generate samples that are used to approximate the posterior distribution of true values of parameters using a Markov Chain Monte Carlo (MCMC) method. BayesReef is used to quantify uncertainty in  _pyReef-Core_  predictions and estimate parameters in the form of a probability distribution. 

# # MCMC methods
MCMC methods, when applied to numerical system models, are used to approximate the posterior distribution of a parameter of interest by randomly sampling in a probabilistic space via a random walk. The random walk generates a chain of samples that are dependent on the current step, but are independent of the previous steps in the chain. The MCMC random walk is designed to take adequately large samples in regions near the solutions of best fit in order to approximate the distribution of parameter values in a model. _BayesReef_ uses the Metropolis-Hastings algorithm MCMC method, which allows for asymmetrical proposals.

MCMC methods are a trademark of a Bayesian inference approach, which is based on Bayes Rule:

<div align="center"><img width=600 src="http://jim-stone.staff.shef.ac.uk/BookBayes2012/HTML_BayesRulev5EbookHTMLFiles/ops/images/f0023-01.jpg" alt="Flowchart of Bayes' Rule by James V Stone, 2018" title="Flowchart of Bayes' Rule"</img>
</div>
    
This formula encapsulates our _prior_ beliefs about a parameter and the _likelihood_ that a chosen parameter value explains the observed data. Combining the prior and likelihood distributions determine the _posterior distribution_, which tells us the parameter values that maximise the probability of the observed data. 

![Alt Text](http://blog.stata.com/wp-content/uploads/2016/11/animation3.gif)

# # Depth-based and time-based inference

_BayesReef_ estimates the posterior distribution of parameters by comparing different forms of data extracted from _pyReef-Core_ simulations. We have two methods of doing this. 

Data about the **depth structure** or the **time structure** of a core can be used as a basis of comparison. The **depth structure** refers to which coralgal assemblage occurs at a given depth interval. The **time structure** of a core refers to which assemblage (or sediment) is deposited at each time interval over the course of the simulation time.

![Schematic diagram of time- and depth-structure of _BayesReef_ simulations](https://www.overleaf.com/docs/12073748dbgfbsqxwxbn/atts/97374009 "Figure 3")


## Installation

### Local install

The code is available from our github [page](https://github.com/pyReef-model/BayesReef.git) and can be obtained either from this page or using **git**
```
git clone https://github.com/pyReef-model/BayesReef.git
```

Once donwloaded, navigate to the **BayesReef** folder and run the following command:
```
pip install -e /workspace/volume/pyReefCore/
```

## Usage

_BayesReef_ is used from a _python script_ directly with Python 2.7.

