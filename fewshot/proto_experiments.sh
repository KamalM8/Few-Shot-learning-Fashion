#ProtoNet fashion small experiments without STN:

python experiments/proto_nets.py --dataset fashion --k-test 2 --n-test 1 --n-train 1
python experiments/proto_nets.py --dataset fashion --k-test 5 --n-test 1 --n-train 1
python experiments/proto_nets.py --dataset fashion --k-test 15 --n-test 1 --n-train 1
python experiments/proto_nets.py --dataset fashion --k-test 2 --n-test 5 --n-train 5
python experiments/proto_nets.py --dataset fashion --k-test 5 --n-test 5 --n-train 5
python experiments/proto_nets.py --dataset fashion --k-test 15 --n-test 5 --n-train 5

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
