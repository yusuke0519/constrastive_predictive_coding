start=$(date)
base="python run_sacred.py -x T with optim.lr=0.0001 seed=1234 gpu=$1"

# context size
eval $base "method.hidden=1600"
eval $base "method.hidden=1600 method.context=400"
eval $base "method.hidden=1600 method.context=800"
eval $base "method.hidden=1600 method.context=1600"
# hidden
eval $base "method.hidden=400"
eval $base "method.hidden=800"
eval $base "method.hidden=3200"
# K
eval $base "method.hidden=1600 method.context=800 dataset.K=1"
eval $base "method.hidden=1600 method.context=800 dataset.K=3"
eval $base "method.hidden=1600 method.context=800 dataset.K=5"
eval $base "method.hidden=1600 method.context=800 dataset.K=7"
# L
eval $base "method.hidden=1600 method.context=800 dataset.L=9"
eval $base "method.hidden=1600 method.context=800 dataset.L=12"
eval $base "method.hidden=1600 method.context=800 dataset.L=15"
# gru
eval $base "method.hidden=1600 method.context=800 method.num_gru=2"
fin=$(date)
echo "Start: $start"
echo "End: $fin"

