import cv2
from glob import glob
from tqdm import tqdm
import random
from threadpool import ThreadPool, makeRequests

all_list = glob('datasets/train/GT/*') + glob('datasets 2/train/GT/*')
target_size = 128
# there are 3 quality ranges for each img
quality_ranges = [(15, 75)]
# only works for jpg
output_path = 'train_LQ_rg'


def saving(path):
    # assert '.jpg' in path
    img = cv2.imread(path)
    kernel_size = random.randint(3, 11)  # 高斯核的大小（奇数）
    sigmaX = random.uniform(0.1, 3.0)  # 标准差的范围
    kernel_size = 2 * kernel_size + 1
    img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigmaX)
    output_file_path = output_path + '/' + path.split('/')[-1]
    # print(output_file_path)
    cv2.imwrite(output_file_path, img)
    # for qr in quality_ranges:
    #     quality = int(random.random() * (qr[1] - qr[0]) + qr[0])
    #     cv2.imwrite(output_path + '/' + path.split('/')[-1], img,
    #                 [int(cv2.IMWRITE_JPEG_QUALITY), quality]) #.replace('.jpg', '_q%d.jpg' % quality)


with tqdm(total=len(all_list), desc='Resizing images') as pbar:
    def callback(req, x):
        pbar.update()

    t_pool = ThreadPool(12)
    requests = makeRequests(saving, all_list, callback=callback)
    for req in requests:
        t_pool.putRequest(req)
    t_pool.wait()
