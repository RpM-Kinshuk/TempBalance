import os
import itertools
from gputracker.gputracker import get_logger, DispatchThread
os.environ['MKL_THREADING_LAYER'] = 'gnu'

gpus = list(range(8))
gpus = [7]
slide_list = [True, False]

seed = 42
slide_window = True
row_samples = 100
q_ratio = 2.0
step_size = 10
sampling_ops = 10

row_list = [20, 50, 100, 150, 200]
qr_list = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5]
ops_list = [5, 10, 15, 20, 30]
seed_list = [42, 43, 44]

dataset = 'cifar100'

grid = itertools.product(seed_list, row_list, qr_list, ops_list)

if not slide_window:
    grid = itertools.product(seed_list, [1], [1], [1])

model = 'resnet'
depth = 18
cachedir = "/scratch/kinshuk/cache"
logger = get_logger('log', 'schedule_subspace.log')

# Bash command list
BASH_COMMAND_LIST = []

for seed, row_samples, q_ratio, sampling_ops in grid:
    
    save_path = f"/jumbo/yaoqingyang/kinshuk/TempBalance/results/flatten/{seed}/{model}{depth}/{dataset}/slide_{slide_window}"
    additional = f"/row_{row_samples}/qr_{q_ratio}/ops_{sampling_ops}"
    if slide_window:
        save_path += additional
    # save_path  = f"/jumbo/yaoqingyang/kinshuk/TempBalance/mew"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    training_stats_path = os.path.join(save_path, 'training_stats.npy')
    if os.path.exists(training_stats_path):
        continue
    cmd = (
        "OMP_NUM_THREADS=1 python /jumbo/yaoqingyang/kinshuk/TempBalance/main_tb.py"
        f" --ckpt-path {save_path}"
        f" --seed {seed}"
        f" --net-type {model}"
        f" --dataset {dataset}"
        f" --lr 0.1"
        f" --weight-decay 0.0005"
        f" --sg 0"
        f" --depth {depth}"
        f" --num-epochs 200"
        f" --batch-size 128"
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
    gpu_m_th=30000,
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