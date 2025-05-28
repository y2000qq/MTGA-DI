import os
import sys
import time
import argparse
import logging


def gpu_info(gpu_id='0,1,2,3'):
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
    while True:
        power_draw, memory_used = gpu_info()

        print('GPUs功耗: ', power_draw)
        print('GPUs内存: ', memory_used)
        logging.info('GPUs功耗: {}'.format(power_draw))
        logging.info('GPUs内存: {}'.format(memory_used))
        time.sleep(interval)

if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logfile = 'GPU_watching.log'
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=logfile)

    logging.info('GPU watching is running...')
    narrow_setup()


