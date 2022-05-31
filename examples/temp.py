import os
import numpy as np
import shutil


data_root = 'F:/work/DATA/boat_reid/images/test'
images = os.listdir(data_root)
# pids = np.arange(0, 107)
# test_pids = np.random.choice(pids, 29, replace=False)
test_pids = [39, 20, 25, 21, 74, 93, 84, 10, 86, 88, 19, 106, 58, 2,
             101, 48, 91, 83, 87, 38, 94, 69, 98, 73, 61, 8, 68, 23, 31]
cam_ids = [1, 2, 3, 4, 5, 6]
select_imgs = []
for pid in test_pids:
    pid_imgs = []
    for img in images:
        pid_ = int(img.split('_')[0])
        if pid_ == pid:
            pid_imgs.append(img)
    indexes = np.arange((len(pid_imgs)))
    select_index = list(np.random.choice(indexes, 4, replace=False))
    for index in select_index:
        select_imgs.append(pid_imgs[index])
for img in images:
    if img in select_imgs:
        shutil.copy(os.path.join(data_root, img), os.path.join('F:/work/DATA/boat_reid/images/query', img))
    else:
        shutil.copy(os.path.join(data_root, img), os.path.join('F:/work/DATA/boat_reid/images/bounding_box_test', img))

print(list(test_pids))


