import matplotlib.pyplot as plt
import os
import datetime

from lib import lib_FDM as fdm



full_path = os.path.realpath(__file__)
_dir,_file=os.path.split(full_path)

dt = datetime.datetime.now()
_date = dt.strftime("%Y%m%d")

init_x=1.0  # 初期位相
init_v=0.0  # 初期速度
init_t=0.0  # 初期時刻
gamma=2.0   # ダンパーの減衰係数
omega=20.0  # 固有角周波数
dt=0.00005  # 時間ステップ
T=1.0       # 合計シミュレーション時間

# FDMによる減衰振動の解を取得
ts, FDM_x, FDM_v, Analytical_x, diff = fdm.FDM_dumper_sim(init_x, init_v, init_t, gamma, omega ,dt, T)

# プロット作成
plt.plot(ts, FDM_x, label='FDM Result')
plt.plot(ts, Analytical_x, label='Analytical solution Result')
plt.title('FDM Solution (dt = {})'.format(dt))
plt.xlabel("time")
plt.ylabel("displacement")
plt.legend()
PATH = _dir+'/data/FDMsol_dt5e-5.png'
plt.savefig(PATH)
plt.show()
    
'''
条件変更
dt : 0.00005 → 0.01 
時間ステップを粗くすると解析解とFDMに不一致が発生
'''

init_x=1.0  # 初期位相
init_v=0.0  # 初期速度
init_t=0.0  # 初期時刻
gamma=2.0   # ダンパーの減衰係数
omega=20.0  # 固有角周波数
dt=0.01     # 時間ステップ
T=1.0       # 合計シミュレーション時間

# FDMによる減衰振動の解を取得
ts, FDM_x, FDM_v, Analytical_x, diff = fdm.FDM_dumper_sim(init_x, init_v, init_t, gamma, omega ,dt, T)

# プロット作成
plt.plot(ts, FDM_x, label='FDM Result')
plt.plot(ts, Analytical_x, label='Analytical solution Result')
plt.title('FDM Solution (dt = {})'.format(dt))
plt.xlabel("time")
plt.ylabel("displacement")
plt.legend()
PATH = _dir+'/data/FDMsol_dt001.png'
plt.savefig(PATH)
plt.show()