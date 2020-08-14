# No augmentation
# Matching networks experiments (fce False) without STN:
#python experiments/matching_nets --dataset fashion --fce False --k-test 2 --n-test 1 --distance l2 --k-train 40
#python experiments/matching_nets.py --dataset fashion --fce False --k-test 5 --n-test 1 --distance l2 --k-train 40
python experiments/matching_nets.py --dataset fashion --fce False --k-test 15 --n-test 1 --distance l2 --k-train 40
python experiments/matching_nets.py --dataset fashion --fce False --k-test 2 --n-test 5 --distance l2 --k-train 40
python experiments/matching_nets.py --dataset fashion --fce False --k-test 5 --n-test 5 --distance l2 --k-train 40
python experiments/matching_nets.py --dataset fashion --fce False --k-test 15 --n-test 5 --distance l2 --k-train 40
exit
