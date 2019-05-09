start=$(date)
base="python run_sacred.py -x T with optim.lr=0.0001 seed=$2 gpu=$1"
omp="OMP_NUM_THREADS=4"

classify="python run_sacred_label.py -x T with optim.lr=0.0001 seed=$2 gpu=$1"
af="classifier.pretrain=True classifier.finetune_g=True classifier.use_c_enc=False classifier.finetune_c=False classifier.auxiliary=1.0"
afgc="classifier.pretrain=True classifier.finetune_g=True classifier.use_c_enc=True classifier.finetune_c=True classifier.auxiliary=1.0"
as="classifier.pretrain=False classifier.finetune_g=True classifier.use_c_enc=False classifier.finetune_c=False classifier.auxiliary=1.0"
asgc="classifier.pretrain=False classifier.finetune_g=True classifier.use_c_enc=True classifier.finetune_c=True classifier.auxiliary=1.0"
cuda="CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=8"

zikkou() {
    eval $omp $base $1 'method.sampler_mode="same"'
    eval $cuda $classify $1 "method.sampler_mode='same'" $af
    eval $cuda $classify $1 "method.sampler_mode='same'" $afgc
    eval $cuda $classify $1 "method.sampler_mode='same'" $as
    eval $cuda $classify $1 "method.sampler_mode='same'" $asgc
    
    # eval $omp $base $1 'method.sampler_mode="diff"'
    # eval $cuda $classify $1 "method.sampler_mode='diff'" $af
    # eval $cuda $classify $1 "method.sampler_mode='diff'" $afgc
    # eval $cuda $classify $1 "method.sampler_mode='diff'" $as
    # eval $cuda $classify $1 "method.sampler_mode='diff'" $asgc
}
# context size
zikkou "method.hidden=1600 method.context=800"
# zikkou "method.hidden=1600 method.context=1600"

# hidden
# zikkou "method.hidden=800"
# zikkou "method.hidden=3200"

fin=$(date)
echo "Start: $start"
echo "End: $fin"

