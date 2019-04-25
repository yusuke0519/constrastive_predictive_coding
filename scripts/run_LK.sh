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
    eval $omp $base $1
    
    eval $cuda $classify $1 $pre
    eval $cuda $classify $1 $fine
    eval $cuda $classify $1 $sup
    eval $cuda $classify $1 $rand
    eval $cuda $classify $1 $prec
    eval $cuda $classify $1 $finec
    eval $cuda $classify $1 $fineg
    eval $cuda $classify $1 $finegc
    eval $cuda $classify $1 $supc
    eval $cuda $classify $1 $supg
    eval $cuda $classify $1 $supgc
}
# context size
zikkou "dataset.L=5"
zikkou "dataset.L=9"
zikkou "dataset.L=15"

# hidden
zikkou "dataset.K=1"
zikkou "dataset.K=3"
zikkou "dataset.K=7"

fin=$(date)
echo "Start: $start"
echo "End: $fin"

