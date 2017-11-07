initmode='in out inout';
inittype='Normal Uniform';
for i in ${initmode}; do
    for j in ${inittype}; do
        screen -S "SELU_${j}_${i}" -d -m bash -c "source activate mlp; nice -n 10 python mnist_task.py --layer_type SELU --layer_num 4 --init_mode $i --init_type $j; exec bash"
    done
done
