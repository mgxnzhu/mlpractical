all:emnist_task.so emnist_task.py run_emnist_task.py
emnist_task.so:emnist_task.c
	gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/afs/inf.ed.ac.uk/user/s17/s1749267/miniconda3/envs/mlp/include/python3.6m -I/afs/inf.ed.ac.uk/user/s17/s1749267/miniconda3/envs/mlp/include/python3.6m  -o emnist_task.so emnist_task.c
emnist_task.c:emnist_task.pyx
	cython emnist_task.pyx
emnist_task.py:emnist_task.pyx
	cp emnist_task.pyx emnist_task.py
	echo "emnist_task()" >> emnist_task.py
run_emnist_task.py:emnist_task.so
	echo "import emnist_task" > run_emnist_task.py
	echo "emnist_task.emnist_task()" >> run_emnist_task.py
