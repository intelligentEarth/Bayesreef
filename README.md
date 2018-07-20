# _BayesReef_: A Bayesian inference framework for _pyReef-Core_ using MCMC methods

## Overview
![Flowchart of BayesReef process. A fusion of multiple sources of data is used to create prior probability distributions on free parameters. Following this, BayesReef is initiated with a vector of free parameters are drawn from the prior. The MCMC sampler uses a Metropolis-Hastings (M-H) algorithm as a basis to accept or reject proposed samples. The sampler terminates when all the allocated samples have been assessed.](figures/BayesReef-MCMC-flowchart.png)


_BayesReef_ is a framework for implementing Bayesian inference on the deterministic model, _pyReef-Core_. It simulates data from the _pyReef-Core_ stratigraphic forward model (SFM) to generate samples which are used to approximate the posterior distribution parameters using a Markov Chain Monte Carlo (MCMC) method. _BayesReef_ is used to quantify uncertainty in  _pyReef-Core_  predictions and estimate parameters in the form of a probability distribution. 

_BayesReef_ is capable of inference of reef core data from the perspective of the depth structure and the time structure of a core. Studies have shown that the software produces accurate parameter estimation and mean model prediction for two- and four-dimensional problems with _pyReef-Core_ (Pall et al., unpublished).

### MCMC methods
MCMC methods, when applied to numerical system models, are used to approximate the posterior distribution of a parameter of interest by randomly sampling in a probabilistic space via a random walk. The random walk generates a chain of samples that are dependent on the current step, but are independent of the previous steps in the chain. The MCMC random walk is designed to take adequately large samples in regions near the solutions of best fit in order to approximate the distribution of parameter values in a model. _BayesReef_ uses the Metropolis-Hastings algorithm MCMC method, which allows for asymmetrical proposals.

MCMC methods are a trademark of a Bayesian inference approach, which is based on Bayes Rule:

<img src="https://cdn-images-1.medium.com/max/1600/1*LB-G6WBuswEfpg20FMighA.png" width="400px">

![Schematic representation of Bayes' rule by Stone, 2013](http://jim-stone.staff.shef.ac.uk/BookBayes2012/HTML_BayesRulev5EbookHTMLFiles/ops/images/f0023-01.jpg?raw=true)

This formula encapsulates our _prior_ beliefs about a parameter and the _likelihood_ that a chosen parameter value explains the observed data. Combining the prior and likelihood distributions determine the _posterior distribution_, which tells us the parameter values that maximise the probability of the observed data. 

To visualise the Metropolis-Hastings MCMC technique, an animation of an example is provided below (Huber, 2016).

![Alt Text](http://blog.stata.com/wp-content/uploads/2016/11/animation3.gif)

### Depth-based and time-based inference

_BayesReef_ estimates the posterior distribution of parameters by comparing different forms of data extracted from _pyReef-Core_ simulations. We have two methods of doing this. 

Data about the **depth structure** or the **time structure** of a core can be used as a basis of comparison. The **depth structure** refers to which coralgal assemblage occurs at a given depth interval. The **time structure** of a core refers to which assemblage (or sediment) is deposited at each time interval over the course of the simulation time. Both perspectives on a reef drill core are presented in the figure below.

![(A) _pyReef-Core_ model output, including the time- and depth-structure of a example drill core. (B) A schematic of modern coral zonation with depth on a forereef representing a shallowing-upward growth strategy, displaying associated assemblage compositions and transitions, adapted from Dechnik (2016)](figures/synthetic_core.png)

## Installation

The code is available from our github [page](https://github.com/pyReef-model/BayesReef.git) and can be obtained either from this page or using **git clone**
```
git clone https://github.com/pyReef-model/BayesReef.git
```
Download the prerequisite python packages provided in the ***installation.txt*** file.

Once donwloaded, navigate to the ***BayesReef*** folder and run the following command:
```
pip install -e /workspace/volume/pyReefCore/
```

## Usage

_BayesReef_ is initiated from python scripts directly with Python 2.7.


***multinom_mcmc_t.py*** and ***multinom_mcmc_d.py*** are the main MCMC scripts from which to run BayesReef. There are 27 free parameters in these file, which includes:

* Population dynamics parameters
  * Malthusian parameter &epsilon; 
  * The Assemblage Interaction Matrix (AIM) parameters &alpha;<sub>m</sub> and  &alpha;<sub>s</sub>
* Hydrodynamic flow exposure threshold parameters (for each assemblage) 
  * _f_<sub>flow</sub><sup>1</sup>
  * _f_<sub>flow</sub><sup>2</sup>
  * _f_<sub>flow</sub><sup>3</sup>
  * _f_<sub>flow</sub><sup>4</sup>
* Sediment imput exposure threshold parameters (for each assemblage)
  * _f_<sub>sed</sub><sup>1</sup>
  * _f_<sub>sed</sub><sup>2</sup>
  * _f_<sub>sed</sub><sup>3</sup>
  * _f_<sub>sed</sub><sup>4</sup>


With alteration, the code is capable of parameterising many other _pyReef-Core_ variables. 


***multinom_mcmc_t_constrained.py*** and ***multinom_mcmc_d_constrained.py*** are similar to the main MCMC scripts, but offer lesser free parameters (up to 4) to allow for low-dimensional experiments with _BayesReef_.


***input_synth.xml*** defines the fixed parameters in _pyReef-Core_. For a greater discussion on _pyReef-Core_, visit the Github repository [here](https://github.com/pyReef-model/pyReefCore).

***1DLikSrf_glv.py***  creates a figure of marginal likelihood (1-D) of population dynamics parameters.

***1DLikSrf_thres.py***  creates a figure of marginal likelihood (1-D) of environmental threshold parameters.

***3DLikSrf_glv.py***  creates a figure of bivariate likelihood (3-D) of population dynamics parameters.

***1DLikSrf_thres.py***  creates a figure of bivariate likelihood (3-D) of environmental threshold parameters.

***runModel.py*** simulates a iteration of _pyReef-Core_ without the Bayesian inference framework. 

## Sample Output

<div align="center">
  <img width=400 src="https://github.com/pyReef-model/BayesReef/blob/master/Testing/time-based_likl/manuscript_results_t/results-constrained-t_25/2p-t-ay.png" title="Density histogram of the posterior distribution of the sub-diagonal parameter." </img> 
  <img width=400 src="https://github.com/pyReef-model/BayesReef/blob/master/Testing/time-based_likl/manuscript_results_t/results-constrained-t_25/2p-t-malth.png" title="Density histogram of the posterior distribution of the Malthusian parameter." </img> 
  <img width=400 src="https://github.com/pyReef-model/BayesReef/blob/master/Testing/time-based_likl/manuscript_results_t/results-constrained-t_25/evol_likl.png" title="Likelihood evolution of two-parameter experiment with 10,000 samples." </img> 
  <img width=400 src="https://github.com/pyReef-model/BayesReef/blob/master/Testing/time-based_likl/manuscript_results_t/results-3d-glv_7/2p-3d-t.png" title="Bivariate posterior likelihood of the Malthusian parameter and sub-diagonal assemblage interaction matrix parameter." </img> 
  <img width=400 src="https://github.com/pyReef-model/BayesReef/blob/master/Testing/time-based_likl/manuscript_results_t/results-constrained-t_25/initpred.png" title="Initial prediction of drill-core data." </img> 
  <img width=400 src="https://github.com/pyReef-model/BayesReef/blob/master/Testing/time-based_likl/manuscript_results_t/results-constrained-t_25/proposals.png" title="All accepted proposals (i.e. predictions) of drill-core data" </img> 
  <img width=400 src="https://github.com/pyReef-model/BayesReef/blob/master/Testing/time-based_likl/manuscript_results_t/results-constrained-t_25/2p-t-mcmcres.png" title="Mean prediction and uncertainty after 20,000 samples in MCMC sampler." </img> 
</div>

## License


his program is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with this program. If not, see http://www.gnu.org/licenses/lgpl-3.0.en.html.

## Reporting

If you come accross a bug or if you need some help compiling or using the code you can drop us a line at: - jodierae.pall@gmail.com and rohitash.chandra@sydney.edu.au

## References

Dechnik, B. (2016) _Evolution of the Great Barrier Reef over the last 130 ka: a multifaceted approach integrating palaeo ecological, palaeo environmental and chronological data from cores_ (Unpublished doctoral thesis). University of Sydney, Sydney, Australia 2006.

Huber, C. (2016). _Introduction to Bayesian statistics, part 2: MCMC and the Metropolis–Hastings algorithm._ [Blog] The Stata Blog. Available at: https://blog.stata.com/2016/11/15/introduction-to-bayesian-statistics-part-2-mcmc-and-the-metropolis-hastings-algorithm/ [Accessed 19 Jul. 2018].

Pall, J., Chandra, R., Azam, D., Salles, T., Webster, J.M. and Cripps, S. (unpublished). _BayesReef: A Bayesian inference framework for modelling reef growth in response to environmental change and biological dynamics._ 

Stone, James. (2013). Bayes' Rule: A Tutorial Introduction to Bayesian Analysis. 10.13140/2.1.1371.6801. 

