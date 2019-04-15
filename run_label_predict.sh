start=$(date)
base="python run_sacred_label.py -x T with optim.lr=0.0001 seed=$2 gpu=$1"
cpc="classifier.pretrain=True classifier.finetune_g=False classifier.use_c_enc=False classifier.finetune_c=False"
cpcf="classifier.pretrain=True classifier.finetune_g=True classifier.use_c_enc=False classifier.finetune_c=False"
sup="classifier.pretrain=False classifier.finetune_g=True classifier.use_c_enc=False classifier.finetune_c=False"
random="classifier.pretrain=False classifier.finetune_g=False classifier.use_c_enc=False classifier.finetune_c=False"
cpcc="classifier.pretrain=True classifier.finetune_g=False classifier.use_c_enc=True classifier.finetune_c=False"
supc="classifier.pretrain=False classifier.finetune_g=True classifier.use_c_enc=True classifier.finetune_c=False"
cuda="CUDA_VISIBLE_DEVICES=$1"

zikkou() {
    eval $cuda $base $1 $cpc
    eval $cuda $base $1 $cpcf
    eval $cuda $base $1 $sup
    eval $cuda $base $1 $random
    eval $cuda $base $1 $cpcc
    eval $cuda $base $1 $supc
}

# context size
zikkou "method.hidden=1600"
zikkou "method.hidden=1600 method.context=400"
zikkou "method.hidden=1600 method.context=800"
zikkou "method.hidden=1600 method.context=1600"
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

