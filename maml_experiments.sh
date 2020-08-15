# maml experiments no augmentation

python -m experiments.maml --dataset fashion --order 1 --n 1 --k 2 --q 5 --meta-batch-size 4 \
    --inner-train-steps 5 --inner-val-steps 10 --inner-lr 0.01 --eval-batches 40 --epoch-len 400
python -m experiments.maml --dataset fashion --order 1 --n 1 --k 5 --q 5 --meta-batch-size 4 \
    --inner-train-steps 5 --inner-val-steps 10 --inner-lr 0.01 --eval-batches 40 --epoch-len 400
python -m experiments.maml --dataset fashion --order 1 --n 1 --k 15 --q 5 --meta-batch-size 4 \
    --inner-train-steps 5 --inner-val-steps 10 --inner-lr 0.01 --eval-batches 40 --epoch-len 400
python -m experiments.maml --dataset fashion --order 1 --n 5 --k 2 --q 5 --meta-batch-size 4 \
    --inner-train-steps 5 --inner-val-steps 10 --inner-lr 0.01 --eval-batches 40 --epoch-len 400
python -m experiments.maml --dataset fashion --order 1 --n 5 --k 5 --q 5 --meta-batch-size 4 \
    --inner-train-steps 5 --inner-val-steps 10 --inner-lr 0.01 --eval-batches 40 --epoch-len 400
python -m experiments.maml --dataset fashion --order 1 --n 5 --k 15 --q 5 --meta-batch-size 4 \
    --inner-train-steps 5 --inner-val-steps 10 --inner-lr 0.01 --eval-batches 40 --epoch-len 400

# maml experiments augmentation

python -m experiments.maml --dataset fashion --order 1 --n 1 --k 2 --q 5 --meta-batch-size 4 --epochs 80 \
    --inner-train-steps 5 --inner-val-steps 10 --inner-lr 0.01 --eval-batches 40 --epoch-len 400 --augment
python -m experiments.maml --dataset fashion --order 1 --n 1 --k 5 --q 5 --meta-batch-size 4 --epochs 80 \
    --inner-train-steps 5 --inner-val-steps 10 --inner-lr 0.01 --eval-batches 40 --epoch-len 400 --augment
python -m experiments.maml --dataset fashion --order 1 --n 1 --k 15 --q 5 --meta-batch-size 4 --epochs 80 \
    --inner-train-steps 5 --inner-val-steps 10 --inner-lr 0.01 --eval-batches 40 --epoch-len 400 --augment
python -m experiments.maml --dataset fashion --order 1 --n 5 --k 2 --q 5 --meta-batch-size 4 --epochs 80 \
    --inner-train-steps 5 --inner-val-steps 10 --inner-lr 0.01 --eval-batches 40 --epoch-len 400 --augment
python -m experiments.maml --dataset fashion --order 1 --n 5 --k 5 --q 5 --meta-batch-size 4 --epochs 80 \
    --inner-train-steps 5 --inner-val-steps 10 --inner-lr 0.01 --eval-batches 40 --epoch-len 400 --augment
python -m experiments.maml --dataset fashion --order 1 --n 5 --k 15 --q 5 --meta-batch-size 4 --epochs 80 \
    --inner-train-steps 5 --inner-val-steps 10 --inner-lr 0.01 --eval-batches 40 --epoch-len 400 --augment
