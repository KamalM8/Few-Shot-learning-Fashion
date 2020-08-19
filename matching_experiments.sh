# No augmentation
# Matching networks experiments (fce False) without augmentation:
python experiments/matching_nets.py --dataset fashion --fce False --k-test 2 --n-test 1 --distance l2 --k-train 20
python experiments/matching_nets.py --dataset fashion --fce False --k-test 5 --n-test 1 --distance l2 --k-train 20
python experiments/matching_nets.py --dataset fashion --fce False --k-test 15 --n-test 1 --distance l2 --k-train 20
python experiments/matching_nets.py --dataset fashion --fce False --k-test 2 --n-test 5 --distance l2 --k-train 20
python experiments/matching_nets.py --dataset fashion --fce False --k-test 5 --n-test 5 --distance l2 --k-train 20
python experiments/matching_nets.py --dataset fashion --fce False --k-test 15 --n-test 5 --distance l2 --k-train 20

# Matching networks experiments (fce True) without augmentation:
python experiments/matching_nets.py --dataset fashion --fce True --k-test 2 --n-test 1 --distance l2 --k-train 20
python experiments/matching_nets.py --dataset fashion --fce True --k-test 5 --n-test 1 --distance l2 --k-train 20
python experiments/matching_nets.py --dataset fashion --fce True --k-test 15 --n-test 1 --distance l2 --k-train 20
python experiments/matching_nets.py --dataset fashion --fce True --k-test 2 --n-test 5 --distance l2 --k-train 20
python experiments/matching_nets.py --dataset fashion --fce True --k-test 5 --n-test 5 --distance l2 --k-train 20
python experiments/matching_nets.py --dataset fashion --fce True --k-test 15 --n-test 5 --distance l2 --k-train 20

# Matching networks experiments (fce False) with augmentation:
python experiments/matching_nets.py --dataset fashion --fce False --k-test 2 --n-test 1 --distance l2 --k-train 20 --augment
python experiments/matching_nets.py --dataset fashion --fce False --k-test 5 --n-test 1 --distance l2 --k-train 20 --augment
python experiments/matching_nets.py --dataset fashion --fce False --k-test 15 --n-test 1 --distance l2 --k-train 20 --augment
python experiments/matching_nets.py --dataset fashion --fce False --k-test 2 --n-test 5 --distance l2 --k-train 20 --augment
python experiments/matching_nets.py --dataset fashion --fce False --k-test 5 --n-test 5 --distance l2 --k-train 20 --augment
python experiments/matching_nets.py --dataset fashion --fce False --k-test 15 --n-test 5 --distance l2 --k-train 20 --augment

# Matching networks experiments (fce True) with augmentation:
python experiments/matching_nets.py --dataset fashion --fce True --k-test 2 --n-test 1 --distance l2 --k-train 20 --augment
python experiments/matching_nets.py --dataset fashion --fce True --k-test 5 --n-test 1 --distance l2 --k-train 20 --augment
python experiments/matching_nets.py --dataset fashion --fce True --k-test 15 --n-test 1 --distance l2 --k-train 20 --augment
python experiments/matching_nets.py --dataset fashion --fce True --k-test 2 --n-test 5 --distance l2 --k-train 20 --augment
python experiments/matching_nets.py --dataset fashion --fce True --k-test 5 --n-test 5 --distance l2 --k-train 20 --augment
python experiments/matching_nets.py --dataset fashion --fce True --k-test 15 --n-test 5 --distance l2 --k-train 20 --augment

# Matching networks experiments (fce False) with STN (coeff: 5.0)
python experiments/matching_nets.py --dataset fashion --fce False --k-test 2 --n-test 1 --distance l2 --k-train 20 --stn_reg_coeff 9.0 --stn 1 --size large --constrained
python experiments/matching_nets.py --dataset fashion --fce False --k-test 5 --n-test 1 --distance l2 --k-train 20 --stn_reg_coeff 9.0 --stn 1 --size large --constrained
python experiments/matching_nets.py --dataset fashion --fce False --k-test 15 --n-test 1 --distance l2 --k-train 20 --stn_reg_coeff 9.0 --stn 1 --size large --constrained
python experiments/matching_nets.py --dataset fashion --fce False --k-test 2 --n-test 5 --distance l2 --k-train 20 --stn_reg_coeff 9.0 --stn 1 --size large --constrained
python experiments/matching_nets.py --dataset fashion --fce False --k-test 5 --n-test 5 --distance l2 --k-train 20 --stn_reg_coeff 9.0 --stn 1 --size large --constrained
python experiments/matching_nets.py --dataset fashion --fce False --k-test 15 --n-test 5 --distance l2 --k-train 20 --stn_reg_coeff 9.0 --stn 1 --size large --constrained

# Matching networks experiments (fce True) with STN (coeff: 5.0)
python experiments/matching_nets.py --dataset fashion --fce True --k-test 2 --n-test 1 --distance l2 --k-train 20 --stn_reg_coeff 9.0 --stn 1 --size large --constrained
python experiments/matching_nets.py --dataset fashion --fce True --k-test 5 --n-test 1 --distance l2 --k-train 20 --stn_reg_coeff 9.0 --stn 1 --size large --constrained
python experiments/matching_nets.py --dataset fashion --fce True --k-test 15 --n-test 1 --distance l2 --k-train 20 --stn_reg_coeff 9.0 --stn 1 --size large --constrained
python experiments/matching_nets.py --dataset fashion --fce True --k-test 2 --n-test 5 --distance l2 --k-train 20 --stn_reg_coeff 9.0 --stn 1 --size large --constrained
python experiments/matching_nets.py --dataset fashion --fce True --k-test 5 --n-test 5 --distance l2 --k-train 20 --stn_reg_coeff 9.0 --stn 1 --size large --constrained
python experiments/matching_nets.py --dataset fashion --fce True --k-test 15 --n-test 5 --distance l2 --k-train 20 --stn_reg_coeff 9.0 --stn 1 --size large --constrained
