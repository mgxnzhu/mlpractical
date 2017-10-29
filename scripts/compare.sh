layertype='Relu Sigmoid LeakyRelu ELU SELU';
for layer in ${layertype}; do
    screen -S "SELU_${layer}" -d -m bash -c "source activate mlp; nice -n 10 python mnist_task.py --layer_type ${layer} --layer_num 2;exec bash"
done
