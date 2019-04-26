
#!/bin/sh 
echo Running all 	
 
# THREE PARAM EXPERIMENT
python mcmc_adapt_gen.py -s 10000 -cs 20 -ci 10 -uc 0 -f 0 
python mcmc_adapt_gen.py -s 10000 -cs 20 -ci 10 -uc 1 -f 0
python mcmc_adapt_gen.py -s 20000 -cs 20 -ci 10 -uc 0 -f 0
python mcmc_adapt_gen.py -s 20000 -cs 20 -ci 10 -uc 1 -f 0

# 11 PARAM EXPERIMENT
python mcmc_adapt_gen.py -s 50000 -cs 20 -ci 10 -uc 0 -f 1
python mcmc_adapt_gen.py -s 50000 -cs 20 -ci 10 -uc 1 -f 1 
python mcmc_adapt_gen.py -s 100000 -cs 20 -ci 10 -uc 1 -f 1 
python mcmc_adapt_gen.py -s 100000 -cs 20 -ci 10 -uc 1 -f 1 

 