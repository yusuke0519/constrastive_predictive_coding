start=$(date)

if [ $1 -eq 0 ]; then
    echo $1
    # context size
    eval "CUDA_VISIBLE_DEVICES=0 python main.py -K 5 -L 12 --hidden 1600 --gru 1"
    eval "CUDA_VISIBLE_DEVICES=0 python main.py -K 5 -L 12 --hidden 1600 --gru 1 --context 800"
    eval "CUDA_VISIBLE_DEVICES=0 python main.py -K 5 -L 12 --hidden 1600 --gru 1 --context 400"
elif [ $1 -eq 1 ]; then
    echo $1
    # hidden size
    eval "CUDA_VISIBLE_DEVICES=1 python main.py -K 5 -L 12 --hidden 400 --gru 1"
    eval "CUDA_VISIBLE_DEVICES=1 python main.py -K 5 -L 12 --hidden 800 --gru 1"
    eval "CUDA_VISIBLE_DEVICES=1 python main.py -K 5 -L 12 --hidden 3200 --gru 1"
elif [ $1 -eq 2 ]; then
    echo $1
    # K
    eval "CUDA_VISIBLE_DEVICES=2 python main.py -K 1 -L 12 --hidden 1600 --gru 1"
    eval "CUDA_VISIBLE_DEVICES=2 python main.py -K 3 -L 12 --hidden 1600 --gru 1"
    eval "CUDA_VISIBLE_DEVICES=2 python main.py -K 7 -L 12 --hidden 1600 --gru 1"
    eval "CUDA_VISIBLE_DEVICES=2 python vae.py -K 3 -L 12 --hidden 1600"
elif [ $1 -eq 3 ]; then
    echo $1
    # L and gru
    eval "CUDA_VISIBLE_DEVICES=3 python main.py -K 5 -L 9 --hidden 1600 --gru 1"
    eval "CUDA_VISIBLE_DEVICES=3 python main.py -K 5 -L 15 --hidden 1600 --gru 1"
    eval "CUDA_VISIBLE_DEVICES=0 python main.py -K 5 -L 12 --hidden 1600 --gru 2"
    eval "CUDA_VISIBLE_DEVICES=2 python vae.py -K 3 -L 12 --hidden 400"
fi
fin=$(date)
echo "Start: $start"
echo "End: $fin"

