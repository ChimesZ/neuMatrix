import numpy as np

def conv_f0 (a, ratio):
    '''
    选取其中`ratio`个连续中最小的
    '''
    f_sum = []
    for i in range(0,len(a)-ratio):
        f = 0
        for j in range(i, i+ratio):
            f = f + a[j]
        f_sum.append(f)
    sum = sorted(f_sum)[0]/ratio
    return sum

def regu (fr,ratio=10):
    for i in range(0,fr.shape[0]):
        a = fr.iloc[i]
        f0 = conv_f0(a,ratio)
        for j in range(0,len(a)):
            a[j] = (a[j]-f0)/f0
        fr.iloc[i] = a
    return fr


def peak(fr,thresh=0.5):
    '''
    现在用的是直接用0.5卡阈值
    '''
    for i in range(0,fr.shape[0]):
        a = fr.iloc[i]
        for j in range(0,len(a)):
            if a[j] > thresh:
                a[j] = 1
            else:
                a[j] = 0
        fr.iloc[i] = a
    return fr

def peak_std(fr,p=3,thresh=0):
    '''
    选取每个细胞均值+p*标准差为阈值，可以同时添加thresh阈值
    '''
    for i in range(0,fr.shape[0]):
        a = fr.iloc[i]
        mean = np.mean(a)
        std = np.std(a)
        for j in range(0,len(a)):
            if a[j] > mean+p*std and a[j] > thresh:
                a[j] = 1
            else:
                a[j] = 0
        fr.iloc[i] = a
    return fr


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range