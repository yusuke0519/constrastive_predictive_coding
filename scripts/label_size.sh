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

zikkou() {
    eval $omp $base $1 'method.sampler_mode="same"'
    
    eval $cuda $classify $1 $2 "method.sampler_mode='same'" $pre
    eval $cuda $classify $1 $2 "method.sampler_mode='same'" $fine
    eval $cuda $classify $1 $2 "method.sampler_mode='same'" $sup
    eval $cuda $classify $1 $2 "method.sampler_mode='same'" $prec
    eval $cuda $classify $1 $2 "method.sampler_mode='same'" $finegc
    eval $cuda $classify $1 $2 "method.sampler_mode='same'" $supgc
    
    eval $cuda $classify $1 $2 "method.sampler_mode='same' classifier.hiddens='[400]'" $pre
    eval $cuda $classify $1 $2 "method.sampler_mode='same' classifier.hiddens='[400]'" $fine
    eval $cuda $classify $1 $2 "method.sampler_mode='same' classifier.hiddens='[400]'" $sup
    eval $cuda $classify $1 $2 "method.sampler_mode='same' classifier.hiddens='[400]'" $prec
    eval $cuda $classify $1 $2 "method.sampler_mode='same' classifier.hiddens='[400]'" $finegc
    eval $cuda $classify $1 $2 "method.sampler_mode='same' classifier.hiddens='[400]'" $supgc
}
# context size
zikkou "method.hidden=1600 method.context=800" "classifier.label_size=0.1"
zikkou "method.hidden=1600 method.context=800" "classifier.label_size=0.2"
zikkou "method.hidden=1600 method.context=800" "classifier.label_size=0.4"

# hidden

fin=$(date)
echo "Start: $start"
echo "End: $fin"

