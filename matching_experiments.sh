# No augmentation
# Matching networks experiments (fce False) without augmentation:
python experiments/matching_nets.py --dataset fashion --fce False --k-test 2 --n-test 1 --distance l2 --k-train 40
python experiments/matching_nets.py --dataset fashion --fce False --k-test 5 --n-test 1 --distance l2 --k-train 40
python experiments/matching_nets.py --dataset fashion --fce False --k-test 15 --n-test 1 --distance l2 --k-train 40
python experiments/matching_nets.py --dataset fashion --fce False --k-test 2 --n-test 5 --distance l2 --k-train 40
python experiments/matching_nets.py --dataset fashion --fce False --k-test 5 --n-test 5 --distance l2 --k-train 40
python experiments/matching_nets.py --dataset fashion --fce False --k-test 15 --n-test 5 --distance l2 --k-train 40

# Matching networks experiments (fce True) without augmentation:
python experiments/matching_nets.py --dataset fashion --fce True --k-test 2 --n-test 1 --distance l2 --k-train 40
python experiments/matching_nets.py --dataset fashion --fce True --k-test 5 --n-test 1 --distance l2 --k-train 40
python experiments/matching_nets.py --dataset fashion --fce True --k-test 15 --n-test 1 --distance l2 --k-train 40
python experiments/matching_nets.py --dataset fashion --fce True --k-test 2 --n-test 5 --distance l2 --k-train 40
python experiments/matching_nets.py --dataset fashion --fce True --k-test 5 --n-test 5 --distance l2 --k-train 40
python experiments/matching_nets.py --dataset fashion --fce True --k-test 15 --n-test 5 --distance l2 --k-train 40

# Matching networks experiments (fce False) with augmentation:
python experiments/matching_nets.py --dataset fashion --fce False --k-test 2 --n-test 1 --distance l2 --k-train 40 --augment
python experiments/matching_nets.py --dataset fashion --fce False --k-test 5 --n-test 1 --distance l2 --k-train 40 --augment
python experiments/matching_nets.py --dataset fashion --fce False --k-test 15 --n-test 1 --distance l2 --k-train 40 --augment
python experiments/matching_nets.py --dataset fashion --fce False --k-test 2 --n-test 5 --distance l2 --k-train 40 --augment
python experiments/matching_nets.py --dataset fashion --fce False --k-test 5 --n-test 5 --distance l2 --k-train 40 --augment
python experiments/matching_nets.py --dataset fashion --fce False --k-test 15 --n-test 5 --distance l2 --k-train 40 --augment

# Matching networks experiments (fce True) with augmentation:
python experiments/matching_nets.py --dataset fashion --fce True --k-test 2 --n-test 1 --distance l2 --k-train 40 --augment
python experiments/matching_nets.py --dataset fashion --fce True --k-test 5 --n-test 1 --distance l2 --k-train 40 --augment
python experiments/matching_nets.py --dataset fashion --fce True --k-test 15 --n-test 1 --distance l2 --k-train 40 --augment
python experiments/matching_nets.py --dataset fashion --fce True --k-test 2 --n-test 5 --distance l2 --k-train 40 --augment
python experiments/matching_nets.py --dataset fashion --fce True --k-test 5 --n-test 5 --distance l2 --k-train 40 --augment
python experiments/matching_nets.py --dataset fashion --fce True --k-test 15 --n-test 5 --distance l2 --k-train 40 --augment
