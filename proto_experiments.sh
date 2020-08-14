#No Augmentation
#ProtoNet fashion small experiments without STN:

python experiments/proto_nets.py --dataset fashion --k-test 2 --n-test 1 --n-train 1 --k-train 40
python experiments/proto_nets.py --dataset fashion --k-test 5 --n-test 1 --n-train 1 --k-train 40
python experiments/proto_nets.py --dataset fashion --k-test 15 --n-test 1 --n-train 1 --k-train 40
python experiments/proto_nets.py --dataset fashion --k-test 2 --n-test 5 --n-train 1 --k-train 40
python experiments/proto_nets.py --dataset fashion --k-test 5 --n-test 5 --n-train 1 --k-train 40
python experiments/proto_nets.py --dataset fashion --k-test 15 --n-test 5 --n-train 1 --k-train 40

#ProtoNet fashion small experiments with STN (coefficent : 10)

python experiments/proto_nets.py --dataset fashion --k-test 2 --n-test 1 --n-train 1 --stn 1
python experiments/proto_nets.py --dataset fashion --k-test 5 --n-test 1 --n-train 1 --stn 1
python experiments/proto_nets.py --dataset fashion --k-test 15 --n-test 1 --n-train 1 --stn 1
python experiments/proto_nets.py --dataset fashion --k-test 2 --n-test 5 --n-train 5 --stn 1
python experiments/proto_nets.py --dataset fashion --k-test 5 --n-test 5 --n-train 5 --stn 1
python experiments/proto_nets.py --dataset fashion --k-test 15 --n-test 5 --n-train 5 --stn 1

#ProtoNet fashion small experiments with STN (coefficent : 0)

python experiments/proto_nets.py --dataset fashion --k-test 2 --n-test 1 --n-train 1 --stn 2
python experiments/proto_nets.py --dataset fashion --k-test 5 --n-test 1 --n-train 1 --stn 2
python experiments/proto_nets.py --dataset fashion --k-test 15 --n-test 1 --n-train 1 --stn 2
python experiments/proto_nets.py --dataset fashion --k-test 2 --n-test 5 --n-train 5 --stn 2
python experiments/proto_nets.py --dataset fashion --k-test 5 --n-test 5 --n-train 5 --stn 2
python experiments/proto_nets.py --dataset fashion --k-test 15 --n-test 5 --n-train 5 --stn 2

#Augmentation
#ProtoNet fashion small experiments without STN:

python experiments/proto_nets.py --dataset fashion --k-test 2 --n-test 1 --n-train 1 --augment
python experiments/proto_nets.py --dataset fashion --k-test 5 --n-test 1 --n-train 1 --augment
python experiments/proto_nets.py --dataset fashion --k-test 15 --n-test 1 --n-train 1 --augment
python experiments/proto_nets.py --dataset fashion --k-test 2 --n-test 5 --n-train 5 --augment
python experiments/proto_nets.py --dataset fashion --k-test 5 --n-test 5 --n-train 5 --augment
python experiments/proto_nets.py --dataset fashion --k-test 15 --n-test 5 --n-train 5 --augment

#ProtoNet fashion small experiments with STN (coefficent : 10)

python experiments/proto_nets.py --dataset fashion --k-test 2 --n-test 1 --n-train 1 --stn 1 --augment
python experiments/proto_nets.py --dataset fashion --k-test 5 --n-test 1 --n-train 1 --stn 1 --augment
python experiments/proto_nets.py --dataset fashion --k-test 15 --n-test 1 --n-train 1 --stn 1 --augment
python experiments/proto_nets.py --dataset fashion --k-test 2 --n-test 5 --n-train 5 --stn 1 --augment
python experiments/proto_nets.py --dataset fashion --k-test 5 --n-test 5 --n-train 5 --stn 1 --augment
python experiments/proto_nets.py --dataset fashion --k-test 15 --n-test 5 --n-train 5 --stn 1 --augment

#ProtoNet fashion small experiments with STN (coefficent : 0)

python experiments/proto_nets.py --dataset fashion --k-test 2 --n-test 1 --n-train 1 --stn 2 --augment
python experiments/proto_nets.py --dataset fashion --k-test 5 --n-test 1 --n-train 1 --stn 2 --augment
python experiments/proto_nets.py --dataset fashion --k-test 15 --n-test 1 --n-train 1 --stn 2 --augment
python experiments/proto_nets.py --dataset fashion --k-test 2 --n-test 5 --n-train 5 --stn 2 --augment
python experiments/proto_nets.py --dataset fashion --k-test 5 --n-test 5 --n-train 5 --stn 2 --augment
python experiments/proto_nets.py --dataset fashion --k-test 15 --n-test 5 --n-train 5 --stn 2 --augment
