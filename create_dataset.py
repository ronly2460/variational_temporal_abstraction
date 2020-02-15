# coding: UTF-8
import numpy as np
import toml
from keras.datasets import mnist


def get_imgs(target_label):
    """
    mnistをロードして、target_labelに対応する画像を取り出す
    """
    (x_train, y_train), (_, _) = mnist.load_data()
    
    files = []
    for label in target_label:
        files.append(x_train[np.where(y_train == label)][0])

    return files


def preprocess(files):
    """
    paddingを行う (28x28 -> 32x32)
    """
    imgs = []

    for img in files:
        img = np.pad(img, [2, 2])
        imgs.append(img.flatten())
                    
    return imgs


def main():
    # parameters
    args = toml.load(open('config.toml'))['dataset']
    
    data_length = args['data_length']
    min_len = args['min_len']
    max_len = args['max_len']
    save_name = args['save_name']
    target_label = args['target_label']

    # start
    files = get_imgs(target_label)
    imgs = preprocess(files)

    # データセット作成
    file_num = len(files)
    cnt = 0
    data = []
    while cnt < data_length:

        #どの数字の画像を使うか
        idx = np.random.choice(file_num)

        # 同じ画像が連続する枚数
        num = np.random.choice(np.arange(min_len, max_len+1))

        # data_lengthを超える場合
        if cnt + num > data_length:
            num = data_length - cnt

        res = np.tile(imgs[idx], num)
        data.append(res)
        cnt = cnt + num

    #reshape (-1, 32, 32) -> (-1, 1, 32, 32)
    data = np.concatenate(data).reshape(-1, 1, 32, 32)
    
    # 保存
    np.save(save_name, data)


if __name__ == '__main__':
    main()