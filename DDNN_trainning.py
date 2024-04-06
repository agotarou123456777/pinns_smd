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



class EarlyStopping:

    def __init__(self, patience=10, verbose=1):
        '''
        Parameters:
            patience(int) : 監視するエポック数
            verbose(int)  : EarlyStopのコメント出力.出力あり(1),出力なし(0)
        '''

        self.epoch = 0 # 監視中のエポック数のカウンター初期化
        self.pre_loss = float('inf') # 比較対象の損失を無限大'inf'で初期化
        self.patience = patience
        self.verbose = verbose

    def __call__(self, current_loss):
        '''
        Parameters:
            current_loss(float): 1エポック終了後の検証データの損失
        Return:
            True:監視回数の上限までに前エポックの損失を超えた場合
            False:監視回数の上限までに前エポックの損失を超えない場合
        '''

        if self.pre_loss < current_loss:
            self.epoch += 1

            if self.epoch > self.patience:
                if self.verbose:
                    print('early stopping')
                return True
            
        else:
            self.epoch = 0
            self.pre_loss = current_loss
        return False


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
解析解から学習用のデータポイントを作成
'''
# Data points
datapoint_list = [i for i in range(0,300,10)] # 学習用データとして抜き出す箇所を設定(0-300要素の間を等間隔(10)で抜き出し)
t_train_data = tf.gather(t, datapoint_list) # タイムステップデータの抜き出し
x_train_data = tf.gather(x, datapoint_list) # 解析解データの抜き出し

plt.scatter(t_train_data, x_train_data, color="lightseagreen", label='training data point') # プロット作成


'''
DDNNの構築と学習
'''
# DDNNのモデル構成、学習パラメータ
n_input = 1    # インプット数
n_output = 1   # アウトプット数
n_neuron = 32  # 隠れ層のユニット数
n_layer = 4    # 隠れ層の層数

plot_interval = 100 # 経過保存用のインターバル(epochインターバル)
epochs = 1000       # エポック数
lr = 1e-3           # learning_rate

# DDNNモデルの構築
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


optimizer = Adam(learning_rate=lr)
loss_fc = MeanSquaredError()
ers = EarlyStopping(patience=50, verbose=True)

model.summary()
'''
model.summary()
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 32)                64        
                                                                 
 dense_1 (Dense)             (None, 32)                1056      
                                                                 
 dense_2 (Dense)             (None, 32)                1056      
                                                                 
 dense_3 (Dense)             (None, 32)                1056      
                                                                 
 dense_4 (Dense)             (None, 1)                 33        
                                                                 
=================================================================
'''

loss_values = [] # DDNNのloss履歴 保存用リスト
x_test_history = [] # DDNNの予測値推移 保存用リスト
t_test_data = np.linspace(0, 1, 20) # テスト用の入力データ作成


# DDNNの学習
for i in range(epochs):
    
    with tf.GradientTape() as tape:
        _pred = model(t_train_data)
        loss = loss_fc(_pred, x_train_data) # モデル予測と正解ラベルから誤差取得
        
    gradients = tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    loss_values.append(loss)
    
    if i % plot_interval == 0:
        print("epoch : ", i)
        x_test_history.append(model(t_test_data))
    
    if ers(loss_values[-1]): #early stoppingの場合ループを抜ける
        break    
    

x_test_data = model.predict(t_test_data) # 学習後モデルを使って、テストデータに対する予測値を取得


# プロット作成
PATH=_dir+'/data/DDNN_prediction.png'
plt.scatter(t_test_data, x_test_data, color='blue', label='DDNN predict result')
plt.title('DDNN prediction')
plt.xlabel("time")
plt.ylabel("displacement")
plt.legend()
plt.savefig(PATH)
plt.show()


PATH=_dir+'/data/DDNN_training_loss.png'
plt.plot(loss_values, label='training') 
plt.ylim(0,0.3)
plt.legend()
plt.title('DDNN Training Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig(PATH)
plt.show()


'''
学習経過のアニメーション作成
'''
imgs = []
fig = plt.figure(figsize=(6, 4))
writerd = animation.PillowWriter(fps=5)
plt.xlabel("time")
plt.ylabel("displacement")

for index, x_test_data in enumerate(x_test_history):
    epoch = index * plot_interval
    text_epoch = "epoch : " + str(epoch)
    text_epoch = fig.text(0.1, 0.9, text_epoch, size = 12, color = "black", fontweight="bold")
    img_1 = plt.plot(t, x, label='analytic solution', color='darkorange')
    img_2 = plt.plot(t_train_data, x_train_data, label='training data point', color="lightseagreen", linestyle='None', marker='o')
    img_3 = plt.plot(t_test_data, x_test_data, label='DDNN prediction', color="blue", linestyle='None', marker='o')
    plt.legend(['analytic solution', 'training data point', 'DDNN prediction'], loc='upper right')
    imgs.append(img_1 + img_2 + img_3 + [text_epoch])

PATH=_dir+'/data/DDNN_Training_Animation.gif'
ani = animation.ArtistAnimation(fig, imgs, interval=100)
ani.save(PATH, writer = animation.PillowWriter(fps=5)) 


'''
モデルの保存
'''
MODEL_PATH=_dir+'/data/model'+'/DDNN/DDNN'
model.save_weights(MODEL_PATH)