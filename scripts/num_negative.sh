start=$(date)
base="python run_sacred.py -x T with optim.lr=0.0001 seed=$2 gpu=$1"
omp="OMP_NUM_THREADS=4"

classify="python run_sacred_label.py -x T with optim.lr=0.0001 seed=$2 gpu=$1"
pre="classifier.pretrain=True classifier.finetune_g=False classifier.use_c_enc=False classifier.finetune_c=False"
fine="classifier.pretrain=True classifier.finetune_g=True classifier.use_c_enc=False classifier.finetune_c=False"
sup="classifier.pretrain=False classifier.finetune_g=True classifier.use_c_enc=False classifier.finetune_c=False"
rand="classifier.pretrain=False classifier.finetune_g=False classifier.use_c_enc=False classifier.finetune_c=False"
prec="classifier.pretrain=True classifier.finetune_g=False classifier.use_c_enc=True classifier.finetune_c=False"
finec="classifier.pretrain=True classifier.finetune_g=False classifier.use_c_enc=True classifier.finetune_c=True"
fineg="classifier.pretrain=True classifier.finetune_g=True classifier.use_c_enc=True classifier.finetune_c=False"
finegc="classifier.pretrain=True classifier.finetune_g=True classifier.use_c_enc=True classifier.finetune_c=True"
supgc="classifier.pretrain=False classifier.finetune_g=True classifier.use_c_enc=True classifier.finetune_c=True"
supg="classifier.pretrain=False classifier.finetune_g=True classifier.use_c_enc=True classifier.finetune_c=False"
supc="classifier.pretrain=False classifier.finetune_g=False classifier.use_c_enc=True classifier.finetune_c=True"
cuda="CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=8"

pretrain() {
    eval $omp $base $1 $2 'method.sampler_mode="same"'
    eval $omp $base $1 $2 'method.sampler_mode="diff"'
}
train() {
    eval $cuda $classify $1 $2 "method.sampler_mode='same'" $pre
    eval $cuda $classify $1 $2 "method.sampler_mode='same'" $prec
    eval $cuda $classify $1 $2 "method.sampler_mode='same'" $fine
    eval $cuda $classify $1 $2 "method.sampler_mode='same'" $finegc
    
    eval $cuda $classify $1 $2 "method.sampler_mode='diff'" $pre
    eval $cuda $classify $1 $2 "method.sampler_mode='diff'" $prec
    eval $cuda $classify $1 $2 "method.sampler_mode='diff'" $fine
    eval $cuda $classify $1 $2 "method.sampler_mode='diff'" $finegc
}

# CPC train
pretrain "method.hidden=1600 method.context=800" "method.num_negative=2"
pretrain "method.hidden=1600 method.context=800" "method.num_negative=4"
pretrain "method.hidden=1600 method.context=800" "method.num_negative=1 method.mask_size=0.2 method.num_mask=10"
pretrain "method.hidden=1600 method.context=800" "method.num_negative=1 method.mask_size=0.2 method.num_mask=5"
pretrain "method.hidden=1600 method.context=800" "method.num_negative=1 method.mask_size=0.2 method.num_mask=2"
pretrain "method.hidden=1601 method.context=800" "method.num_negative=1 method.mask_size=0.1 method.num_mask=10"
pretrain "method.hidden=1600 method.context=800" "method.num_negative=1 method.mask_size=0.1 method.num_mask=20"
pretrain "method.hidden=1600 method.context=800" "method.num_negative=1 method.mask_size=0.05 method.num_mask=10"

# Classifier train
train "method.hidden=1600 method.context=800" "method.num_negative=2"
train "method.hidden=1600 method.context=800" "method.num_negative=4"
train "method.hidden=1600 method.context=800" "method.num_negative=1 method.mask_size=0.2 method.num_mask=10"
train "method.hidden=1600 method.context=800" "method.num_negative=1 method.mask_size=0.2 method.num_mask=5"
train "method.hidden=1600 method.context=800" "method.num_negative=1 method.mask_size=0.2 method.num_mask=2"
train "method.hidden=1600 method.context=800" "method.num_negative=1 method.mask_size=0.1 method.num_mask=10"
train "method.hidden=1600 method.context=800" "method.num_negative=1 method.mask_size=0.1 method.num_mask=20"
train "method.hidden=1600 method.context=800" "method.num_negative=1 method.mask_size=0.05 method.num_mask=10"

fin=$(date)
echo "Start: $start"
echo "End: $fin"

