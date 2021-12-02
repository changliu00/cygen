# MNIST
# * cygen
python3 main_image.py --dataset mnist --method cygen --warm_up 100 --kl_beta 1. --w_cm 1e-3 --vae_clamp 100 --x_gen_stepsize 1e-3 --niters 10
# * vae
python3 main_image.py --dataset mnist --method elbo --warm_up 100
# * dae
python3 main_image.py --dataset mnist --method dae --set_p_var 1.

# SVHN
# * cygen
python3 main_image.py --dataset svhn --method cygen --warm_up 100 --kl_beta 0.01 --w_cm 1e-3 --vae_clamp 0.1 --x_gen_stepsize 1e-3 --niters 20
# * vae
python3 main_image.py --dataset svhn --method elbo --warm_up 100
# * dae
python3 main_image.py --dataset svhn --method dae --set_p_var 1.

