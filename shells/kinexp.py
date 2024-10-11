import os
import itertools
from gputracker.gputracker import get_logger, DispatchThread
os.environ['MKL_THREADING_LAYER'] = 'gnu'

gpus = list(range(4))

slide_window = True
slide_list = [True, False]
row_samples = 100
q_ratio = 2.0
step_size = 10
sampling_ops = 10
qr_list = [0.5, 1.0, 1.5, 2.0]
ops_list = [5, 10, 15, 20]

dataset = 'cifar100'

grid = itertools.product(qr_list, ops_list)

if not slide_window:
    grid = itertools.product([1], [1])

model = 'resnet'
cachedir = "/scratch/kinshuk/cache"
logger = get_logger('log', 'schedule_subspace.log')

# Bash command list
BASH_COMMAND_LIST = []

for q_ratio, sampling_ops in grid:
    
    save_path = f"/jumbo/yaoqingyang/kinshuk/TempBalance/results/{model}18/{dataset}"
    additional = f"/slide_{slide_window}/row_{row_samples}/qr_{q_ratio}/ops_{sampling_ops}"
    save_path += additional
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cmd = (
        "OMP_NUM_THREADS=1 python /jumbo/yaoqingyang/kinshuk/TempBalance/main_tb.py"
        f" --ckpt-path {save_path}"
        f" --seed 42"
        f" --net-type {model}"
        f" --dataset {dataset}"
        f" --lr 0.01"
        f" --depth 18"
        f" --num-epochs 200"
        f" --batch-size 512"
        f" --optim-type SGD"
        f" --pl-fitting median"
        f" --use-tb true"
        f" --assign-func tb_linear_map"
        f" --use-sliding-window {slide_window}"
        f" --row-samples {row_samples}"
        f" --Q-ratio {q_ratio}"
        f" --step-size {step_size}"
        f" --sampling-ops-per-dim {sampling_ops}"
    )

    BASH_COMMAND_LIST.append(cmd)

# Dispatch thread setup
dispatch_thread = DispatchThread(
    "TempBalance training",
    BASH_COMMAND_LIST,
    logger,
    gpu_m_th=500,
    gpu_list=gpus,
    maxcheck=0,
    num_gpus_needed=1,
)

# Start and join the dispatch thread
dispatch_thread.start()
dispatch_thread.join()

import time
time.sleep(5)

logger.info("Exiting Main Thread")