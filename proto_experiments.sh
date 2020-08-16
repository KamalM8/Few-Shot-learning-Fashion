# No Augmentation
# ProtoNet fashion experiments without STN:

python experiments/proto_nets.py --dataset fashion --k-test 2 --n-test 1 --n-train 1 --k-train 20 --size large
python experiments/proto_nets.py --dataset fashion --k-test 5 --n-test 1 --n-train 1 --k-train 20 --size large
python experiments/proto_nets.py --dataset fashion --k-test 15 --n-test 1 --n-train 1 --k-train 20 --size large
python experiments/proto_nets.py --dataset fashion --k-test 2 --n-test 5 --n-train 1 --k-train 20 --size large
python experiments/proto_nets.py --dataset fashion --k-test 5 --n-test 5 --n-train 1 --k-train 20 --size large
python experiments/proto_nets.py --dataset fashion --k-test 15 --n-test 5 --n-train 1 --k-train 20 --size large

# Augmentation
# ProtoNet fashion experiments without STN and with Augmentation:
python experiments/proto_nets.py --dataset fashion --k-test 2 --n-test 1 --n-train 1 --k-train 20 --augment --size large
python experiments/proto_nets.py --dataset fashion --k-test 5 --n-test 1 --n-train 1 --k-train 20 --augment --size large
python experiments/proto_nets.py --dataset fashion --k-test 15 --n-test 1 --n-train 1 --k-train 20 --augment --size large
python experiments/proto_nets.py --dataset fashion --k-test 2 --n-test 5 --n-train 5 --k-train 20 --augment --size large
python experiments/proto_nets.py --dataset fashion --k-test 5 --n-test 5 --n-train 5 --k-train 20 --augment --size large
python experiments/proto_nets.py --dataset fashion --k-test 15 --n-test 5 --n-train 5 --k-train 20 --augment --size large

