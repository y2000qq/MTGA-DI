# modify: Arlse
# author: levio
# contact: levio.pku@gmail.com

import os
import sys
import time
import argparse

'''
GPUs 排队脚本
'''
parser = argparse.ArgumentParser(description='GPU scramble')
parser.add_argument('--cmd', type=str, default='bash ./scripts/train_3091.sh')
parser.add_argument('--gpu', type=str, default='1')
args = parser.parse_args()
print(args)

def gpu_info(gpu_id):
    gpu_id = [int(i) for i in gpu_id.split(',')]
    gpu_status = os.popen(
        'nvidia-smi --query-gpu=index,memory.used,power.draw --format=csv,noheader,nounits').readlines()

    memory_used = []
    power_draw = []
    for info in gpu_status:
        info = info.strip().split(',')
        gpu_index = int(info[0])
        gpu_memory_used = int(info[1])
        gpu_power_draw = int(float(info[2]))
        if gpu_index in gpu_id:
            memory_used.append(gpu_memory_used)
            power_draw.append(gpu_power_draw)

    return power_draw, memory_used

def narrow_setup(interval=10):
    power_draw, memory_used = gpu_info(gpu_id=args.gpu)

    while any(memory > 1000 for memory in memory_used) or any(power > 60 for power in power_draw):  # set waiting condition
        power_draw, memory_used = gpu_info(gpu_id=args.gpu)

        print("GPU_{}'s功耗: ".format(args.gpu), power_draw)
        print("GPU_{}'s内存: ".format(args.gpu), memory_used)
        time.sleep(interval)

    print('GPU_{} are available now!'.format(args.gpu))
    print('\n' + args.cmd)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    time.sleep(120)
    os.system(args.cmd)


if __name__ == '__main__':
    narrow_setup()
