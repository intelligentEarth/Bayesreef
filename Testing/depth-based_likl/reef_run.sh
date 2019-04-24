
#!/bin/sh 
echo Running all 	
 
python mcmc_adapt_gen.py -s 10000 -cs 20 -ci 10 -uc 0 -f 0 &
python mcmc_adapt_gen.py -s 10000 -cs 20 -ci 10 -uc 1 -f 0 &
python mcmc_adapt_gen.py -s 20000 -cs 20 -ci 10 -uc 0 -f 0 &
python mcmc_adapt_gen.py -s 20000 -cs 20 -ci 10 -uc 1 -f 0 &
for x in 1 2 3 
	do
			# python ptBayeslands_sedvec.py -p $prob -s 10000 -r 10 -t 10 -swap 0.01 -b 0.25 -pt 0.5
 			python mcmc_adapt_gen.py -s 50000 -cs 20 -ci 10 -uc 0 -f 1 & 
			python mcmc_adapt_gen.py -s 50000 -cs 20 -ci 10 -uc 1 -f 1 & 			

	done



 