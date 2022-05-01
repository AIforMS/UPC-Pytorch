import numpy as np
from glob import glob
from tqdm import tqdm
import h5py
import nrrd

output_size =[112, 112, 80]     # 裁剪图片成为112*112*80
# 160,192,224
# nnrd包读取后，预处理后转变为h5py文件
def covert_h5():
    listt = glob('E:/project/Master-1/code/segmetation-UAMT/UA-MT/data/Training_Set/*/lgemri.nrrd')
    print(listt)

    for item in tqdm(listt):
        image, img_header = nrrd.read(item)
        label, gt_header = nrrd.read(item.replace('lgemri.nrrd', 'laendo.nrrd'))
        label = (label == 255).astype(np.uint8)
        w, h, d = label.shape

        tempL = np.nonzero(label)  # 得到数组array中非零元素的位置（下标）
        minx, maxx = np.min(tempL[0]), np.max(tempL[0])
        miny, maxy = np.min(tempL[1]), np.max(tempL[1])
        minz, maxz = np.min(tempL[2]), np.max(tempL[2])

        px = max(output_size[0] - (maxx - minx), 0) // 2
        py = max(output_size[1] - (maxy - miny), 0) // 2
        pz = max(output_size[2] - (maxz - minz), 0) // 2
        minx = max(minx - np.random.randint(10, 20) - px, 0)
        maxx = min(maxx + np.random.randint(10, 20) + px, w)
        miny = max(miny - np.random.randint(10, 20) - py, 0)
        maxy = min(maxy + np.random.randint(10, 20) + py, h)
        minz = max(minz - np.random.randint(5, 10) - pz, 0)
        maxz = min(maxz + np.random.randint(5, 10) + pz, d)

        image = (image - np.mean(image)) / np.std(image)  # 标准化/0均值化，将数据中心转移到原点处，np.std计算标准差
        image = image.astype(np.float32)  # 转为浮点数
        image = image[minx:maxx, miny:maxy]
        label = label[minx:maxx, miny:maxy]
        print(label.shape)
        f = h5py.File(item.replace('lgemri.nrrd', 'mri_norm2.h5'), 'w')
        f.create_dataset('image', data=image, compression="gzip")
        f.create_dataset('label', data=label, compression="gzip")
        f.close()


if __name__ == '__main__':
    covert_h5()