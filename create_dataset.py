import numpy as np
from keras.datasets import mnist


def extract_imgs():
    """
    mnistをロードして、0, 1, 3, 7, 9の画像を取り出す
    """
    (x_train, _), (_, _) = mnist.load_data()
    
    zero = x_train[1]
    one = x_train[3]
    three = x_train[7]
    seven = x_train[15]
    nine = x_train[4]

    files = [zero, one, three, seven, nine]
    return files


def preprocess(files):
    """
    paddingを行う (28x28 -> 32x32)
    """
    imgs = np.empty((0, 32, 32))

    for img in files:
        img = np.pad(img, [2, 2])
        imgs = np.vstack((imgs, img))
                    
    return imgs


def main():
    # parameters
    data_length =  100000 # データの個数
    min_len = 3                # 連続する最小枚数
    max_len = 10              # 連続する最大枚数
    save_name = './data/imgs'
    
    files = extract_imgs()
    imgs = preprocess(files)
    
    file_num = len(files)
    cnt = 0
    data = np.empty((0, 32, 32))
    
    # データセット作成
    while cnt < data_length:
        #どの数字の画像を使うか
        idx = np.random.choice(file_num)
        
        # 同じ画像が連続する枚数
        num = np.random.choice(np.arange(min_len, max_len+1))
        
        # data_lengthを超える場合
        if cnt + num > data_length:
            num = data_length - cnt
        
        res = np.array([imgs[idx] for _ in range(num)])
        data = np.vstack((data, res))
        cnt = cnt + num
    
    #reshape (-1, 32, 32) -> (-1, 1, 32, 32)
    data = data.reshape(-1, 1, 32, 32)
    
    # 保存
    np.save(save_name, data)