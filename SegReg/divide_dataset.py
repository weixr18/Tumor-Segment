# divide dataset script
import os
import random

root = "/mnt/data1/mvi2/h5_3mod_512"
files = os.listdir(root)
random.shuffle(files)
with open('train_rs.list', 'w') as f:
    for item in files[:len(files)//2]:
        f.writelines(os.path.join(root, item)+'\n')
with open('test_rs.list', 'w') as f:
    for item in files[len(files)//2:]:
        f.writelines(os.path.join(root, item)+'\n')
