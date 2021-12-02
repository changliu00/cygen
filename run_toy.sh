# pinwheel
python3 main_toy.py --method dae --lr 1e-4 --niters 12000
python3 main_toy.py --method dae --pretr_method elbo --pretr_niter 1000 --lr 1e-4 --niters 12000

python3 main_toy.py --method elbo
python3 main_toy.py --method bigan --pxtype js
python3 main_toy.py --method gibbs --pxtype js --n_gibbs 2 --niters 12000

python3 main_toy.py --method cygen
python3 main_toy.py --method cygen --pretr_method elbo --pretr_niter 1000
python3 main_toy.py --method cygen --pretr_method elbo --pretr_niter 1000 --w_cm 0.

# 8gaussians
python3 main_toy.py --method dae --lr 1e-4 --niters 12000 --data 8gaussians
python3 main_toy.py --method dae --pretr_method elbo --pretr_niter 1000 --lr 1e-4 --niters 12000 --data 8gaussians

python3 main_toy.py --method elbo --data 8gaussians
python3 main_toy.py --method bigan --pxtype js --lr 1e-4 --lr_d 3e-5 --data 8gaussians
python3 main_toy.py --method bigan --pxtype js --n_biggs 2 --niters 12000 --lr 1e-4 --lr_d 3e-5 --data 8gaussians

python3 main_toy.py --method cygen --data 8gaussians
python3 main_toy.py --method cygen --pretr_method elbo --pretr_niter 1000 --data 8gaussians
python3 main_toy.py --method cygen --pretr_method elbo --pretr_niter 1000 --w_cm 0. --data 8gaussians

