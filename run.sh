start=$(date)
base="with optim.lr=0.0001 seed=$2 gpu=$1"
cuda="CUDA_VISIBLE_DEVICES=$1"

zikkou() {
    echo $cuda
    eval "python run_sacred.py -x T" $base $1
}

# context size
zikkou "method.hidden=1600"
zikkou "method.hidden=1600 method.context=400"
zikkou "method.hidden=1600 method.context=800"
zikkou "method.hidden=1600 method.context=1600"

eval "sh run_label_predict.sh" $1 $2
# hidden
zikkou "method.hidden=400"
zikkou "method.hidden=800"
zikkou "method.hidden=3200"
# K
zikkou "method.hidden=1600 method.context=800 dataset.K=1"
zikkou "method.hidden=1600 method.context=800 dataset.K=3"
zikkou "method.hidden=1600 method.context=800 dataset.K=5"
zikkou "method.hidden=1600 method.context=800 dataset.K=7"
# L
zikkou "method.hidden=1600 method.context=800 dataset.L=9"
zikkou "method.hidden=1600 method.context=800 dataset.L=12"
zikkou "method.hidden=1600 method.context=800 dataset.L=15"
# gru
zikkou "method.hidden=1600 method.context=800 method.num_gru=2"
fin=$(date)
echo "Start: $start"
echo "End: $fin"

