# coding: UTF-8
import numpy as np
import toml
from keras.datasets import mnist


def get_imgs(target_label, fix=True, max_len=100):
    """
    mnistをロードして、target_labelに対応する画像を取り出す
    """
    (x_train, y_train), _ = mnist.load_data()
    
    imgs_sets = []
    for label in target_label:
        if fix:
            imgs_sets.append(x_train[np.where(y_train == label)][0])
        else:
            x_train, y_train = x_train[:max_len], y_train[:max_len]
            imgs_sets.append(x_train[np.where(y_train == label)])

    return imgs_sets


def preprocess(imgs_sets, fix=True):
    """
    paddingを行う (28x28 -> 32x32)
    """
    result = []
    for imgs in imgs_sets:
        if fix:
            img = np.pad(imgs, [2, 2])
            result.append(img.flatten())
        else:
            tmp = []
            for img in imgs:
                img = np.pad(img, [2,2])
                tmp.append(img.flatten())
            result.append(np.array(tmp))
                    
    return result


def main():
    # parameters
    args = toml.load(open('config.toml'))['dataset']
    
    data_length = args['data_length']
    min_len = args['min_len']
    max_len = args['max_len']
    save_name = args['save_name']
    target_label = args['target_label']
    fix = args['fix']
    target_num = len(target_label)
    
    # start
    imgs_sets = get_imgs(target_label, fix)
    imgs_sets = preprocess(imgs_sets, fix)

    # データセット作成
    cnt = 0
    data = []
    while cnt < data_length:
        
        # 同じ画像が連続する枚数
        repeat_times = np.random.choice(np.arange(min_len, max_len+1))
        # data_lengthを超える場合
        if cnt + repeat_times > data_length:
            repeat_times = data_length - cnt
        
        # どの数字の画像を使うか
        num = np.random.choice(target_num)
        
        if fix:
            res = np.tile(imgs_sets[num], repeat_times)
        else:
            # ある数字のどの画像を使うか
            idx = np.random.choice(imgs_sets[num].shape[0])
            res = np.tile(imgs_sets[num][idx], repeat_times)
            
        data.append(res)
        cnt = cnt + repeat_times
    
    #reshape (-1, 32, 32) -> (-1, 1, 32, 32)
    data = np.concatenate(data).reshape(-1, 1, 32, 32)
    
    # 保存
    np.save(save_name, data)


if __name__ == '__main__':
    main()