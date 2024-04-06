import tensorflow as tf
from keras.losses import MeanSquaredError
from keras.optimizers import Adam
from keras.layers import Dense
from keras.models import Sequential
from keras.initializers import GlorotNormal

import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

from lib import lib_FDM as fdm


full_path = os.path.realpath(__file__)
_dir,_file=os.path.split(full_path)

'''
解析解を求める
'''
gamma = 2 # ダンパーの減衰係数
omega = 20 # 固有角周波数

t = np.linspace(0,1,500) # タイムステップを生成(0s-1sの間を500分割)
t = np.reshape(t,[-1,1])
x = fdm.analytical_solution(gamma, omega, t) # タイムステップごとの解析解を計算
x = np.reshape(x,[-1,1])

plt.plot(t, x, color='darkorange', label='analytic solution') # プロット


'''
DDNNモデルをロードして結果予測
'''
# DDNNモデルの構築
n_input = 1    # インプット数
n_output = 1   # アウトプット数
n_neuron = 32  # 隠れ層のユニット数
n_layer = 4    # 隠れ層の層数


model = Sequential()

model.add(Dense(units=n_neuron, 
                activation='tanh', 
                kernel_initializer=GlorotNormal(), 
                input_shape=(n_input,)))
    
for i in range(n_layer-1):
    model.add(Dense(units=n_neuron, 
                    activation='tanh', 
                    kernel_initializer=GlorotNormal()))
        
model.add(Dense(units=n_output))      

model.summary()

MODEL_PATH=_dir+'/data/model'+'/DDNN/DDNN'
model.load_weights(MODEL_PATH)

t_test_data = np.linspace(0, 1, 100) # テスト用の入力データ作成
x_test_data = model.predict(t_test_data) # 学習後モデルを使って、テストデータに対する予測値を取得

plt.plot(t_test_data, x_test_data, color='blue', label='DDNN predict result')
plt.legend()
plt.xlabel("time")
plt.ylabel("displacement")
plt.show()

