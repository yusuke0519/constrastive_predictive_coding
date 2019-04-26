start=$(date)
base="python run_sacred.py -x T with optim.lr=0.0001 seed=$2 gpu=$1 method.name='VAE'"
omp="OMP_NUM_THREADS=4"

classify="python run_sacred_label.py -x T with optim.lr=0.0001 seed=$2 gpu=$1 method.name='VAE'"
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
    eval $omp $base $1
    
    eval $cuda $classify $1 $pre
    eval $cuda $classify $1 $fine
    eval $cuda $classify $1 $finec
    eval $cuda $classify $1 $finegc
    
    eval $cuda $classify $1 "classifier.hiddens='[400]'" $pre
    eval $cuda $classify $1 "classifier.hiddens='[400]'" $fine
    eval $cuda $classify $1 "classifier.hiddens='[400]'" $finec
    eval $cuda $classify $1 "classifier.hiddens='[400]'" $finegc
}

labelsize() {
    eval $omp $base $1
    
    eval $cuda $classify $1 $2 $pre
    eval $cuda $classify $1 $2 $fine
    eval $cuda $classify $1 $2 $finec
    eval $cuda $classify $1 $2 $finegc
    
    eval $cuda $classify $1 $2 "classifier.hiddens='[400]'" $pre
    eval $cuda $classify $1 $2 "classifier.hiddens='[400]'" $fine
    eval $cuda $classify $1 $2 "classifier.hiddens='[400]'" $finec
    eval $cuda $classify $1 $2 "classifier.hiddens='[400]'" $finegc
}
# hidden
zikkou "method.hidden=400"
zikkou "method.hidden=800"
zikkou "method.hidden=1600"
zikkou "method.hidden=3200"
# hidden
zikkou "optim.num_batch=30000"

# labelsize
labelsize "method.hidden=1600" "classifier.label_size=0.1"
labelsize "method.hidden=1600" "classifier.label_size=0.2"
labelsize "method.hidden=1600" "classifier.label_size=0.4"

fin=$(date)
echo "Start: $start"
echo "End: $fin"

