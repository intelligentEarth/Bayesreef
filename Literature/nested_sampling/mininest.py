#                   NESTED SAMPLING MAIN PROGRAM
# (GNU General Public License software, (C) Sivia and Skilling 2006)
# This file was translated to Python by Issac Trotts in 2007.

from math import *
import random

# or so
DBL_MAX = 1e300

# ~U[0,1)
uniform = random.random

# logarithmic addition log(exp(x)+exp(y))
def plus(x,y):
    if x>y:
        return x+log(1+exp(y-x)) 
    else:
        return y+log(1+exp(x-y))

# n = number of objects to evolve
def nested_sampling(n, max_iter, sample_from_prior, explore):
    """
    This is an implementation of John Skilling's Nested Sampling algorithm
    for computing the normalizing constant of a probability distribution
    (usually the posterior in Bayesian inference).

    The return value is a dictionary with the following entries:
        "samples"
        "num_iterations"
        "logZ"
        "logZ_sdev"
        "info_nats"
        "info_sdev"

    More information is available here:
    http://www.inference.phy.cam.ac.uk/bayesys/
    """
    # FIXME: Add a simple example to the doc string.

    Obj = []              # Collection of n objects
    Samples = []          # Objects stored for posterior results
    logwidth = None       # ln(width in prior mass)
    logLstar = None       # ln(Likelihood constraint)
    H    = 0.0            # Information, initially 0
    logZ =-DBL_MAX        # ln(Evidence Z, initially 0)
    logZnew = None        # Updated logZ
    copy = None           # Duplicated object
    worst = None          # Worst object
    nest = None           # Nested sampling iteration count

    # Set prior objects
    for i in range(n):
        Obj.append(sample_from_prior())

    # Outermost interval of prior mass
    logwidth = log(1.0 - exp(-1.0 / n));

    # NESTED SAMPLING LOOP ___________________________________________
    for nest in range(max_iter):

        # Worst object in collection, with Weight = width * Likelihood
        worst = 0;
        for i in range(1,n):
            if Obj[i].logL < Obj[worst].logL:
                worst = i

        Obj[worst].logWt = logwidth + Obj[worst].logL;

        # Update Evidence Z and Information H
        logZnew = plus(logZ, Obj[worst].logWt)
        H = exp(Obj[worst].logWt - logZnew) * Obj[worst].logL + \
            exp(logZ - logZnew) * (H + logZ) - logZnew;
        logZ = logZnew;

        # Posterior Samples (optional)
        Samples.append(Obj[worst])

        # Kill worst object in favour of copy of different survivor
        if n>1: # don't kill if n is only 1
            while True:
                copy = int(n * uniform()) % n  # force 0 <= copy < n
                if copy != worst:
                    break

        logLstar = Obj[worst].logL;       # new likelihood constraint
        Obj[worst] = Obj[copy];           # overwrite worst object

        # Evolve copied object within constraint
        updated = explore(Obj[worst], logLstar);
        assert(updated != None) # Make sure explore didn't update in-place
        Obj[worst] = updated

        # Shrink interval
        logwidth -= 1.0 / n;

    # Exit with evidence Z, information H, and optional posterior Samples
    sdev_H = H/log(2.)
    sdev_logZ = sqrt(H/n)
    return {"samples":Samples, 
            "num_iterations":(nest+1), 
            "logZ":logZ,
            "logZ_sdev":sdev_logZ,
            "info_nats":H,
            "info_sdev":sdev_H}

