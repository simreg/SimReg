import os
import shutil
from glob import glob

cleaning_root='../runs'

if __name__ == '__main__':
    exit(0)
    for x in list(glob(f'{cleaning_root}/*/checkpoint-*')):
        if 'mini_bert/' in x:
            continue
        print(f'removing {x}')
        shutil.rmtree(x)
