layernum='2 3 4 5 6 7 8';
for j in ${layernum}; do
    screen -S "SELU_$j" -d -m bash -c "source activate mlp; nice -n 10 python mnist_task.py --layer_type SELU --layer_num $j; exec bash"
done
