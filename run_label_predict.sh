start=$(date)
base="python run_sacred_label.py -x T with optim.lr=0.0001 seed=$2 gpu=$1"
pre="classifier.pretrain=True classifier.finetune_g=False classifier.use_c_enc=False classifier.finetune_c=False"
fine="classifier.pretrain=True classifier.finetune_g=True classifier.use_c_enc=False classifier.finetune_c=False"
sup="classifier.pretrain=False classifier.finetune_g=True classifier.use_c_enc=False classifier.finetune_c=False"
rand="classifier.pretrain=False classifier.finetune_g=False classifier.use_c_enc=False classifier.finetune_c=False"

prec="classifier.pretrain=True classifier.finetune_g=False classifier.use_c_enc=True classifier.finetune_c=False"
finec="classifier.pretrain=True classifier.finetune_g=False classifier.use_c_enc=True classifier.finetune_c=True"
fineg="classifier.pretrain=True classifier.finetune_g=False classifier.use_c_enc=True classifier.finetune_c=True"
finegc="classifier.pretrain=True classifier.finetune_g=True classifier.use_c_enc=True classifier.finetune_c=True"
supgc="classifier.pretrain=False classifier.finetune_g=True classifier.use_c_enc=True classifier.finetune_c=True"
supg="classifier.pretrain=False classifier.finetune_g=True classifier.use_c_enc=True classifier.finetune_c=False"
supc="classifier.pretrain=False classifier.finetune_g=False classifier.use_c_enc=True classifier.finetune_c=True"
cuda="CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=8"

zikkou() {
    eval $cuda $base $1 $pre
    eval $cuda $base $1 $fine
    eval $cuda $base $1 $sup
    eval $cuda $base $1 $rand
    eval $cuda $base $1 $prec
    eval $cuda $base $1 $finec
    eval $cuda $base $1 $fineg
    eval $cuda $base $1 $finegc
    eval $cuda $base $1 $supc
    eval $cuda $base $1 $supg
    eval $cuda $base $1 $supgc
    
    eval $cuda $base $1 "method.sampler_mode='diff'" $pre
    eval $cuda $base $1 "method.sampler_mode='diff'" $fine
    eval $cuda $base $1 "method.sampler_mode='diff'" $prec
    eval $cuda $base $1 "method.sampler_mode='diff'" $finec
    eval $cuda $base $1 "method.sampler_mode='diff'" $fineg
    eval $cuda $base $1 "method.sampler_mode='diff'" $finegc
    
    eval $cuda $base $1 "method.sampler_mode='same'" $pre
    eval $cuda $base $1 "method.sampler_mode='same'" $fine
    eval $cuda $base $1 "method.sampler_mode='same'" $prec
    eval $cuda $base $1 "method.sampler_mode='same'" $finec
    eval $cuda $base $1 "method.sampler_mode='same'" $fineg
    eval $cuda $base $1 "method.sampler_mode='same'" $finegc
}

# context size
zikkou "method.hidden=1600 method.context=800"
zikkou "method.hidden=1600 method.context=1600"

# hidden
zikkou "method.hidden=800"
zikkou "method.hidden=3200"

fin=$(date)
echo "Start: $start"
echo "End: $fin"

