import numpy as np

def analytical_solution(gamma, omega, t):
    '''
    バネ・マス・ダンパー系の減衰振動の解析解を求める関数
    
    Input for Function : 
    gamma   || ダンパーの減衰係数
    omega   || 固有角周波数
    t       || 時間刻
    
    Return values from Function :
    x || 解析解
    '''
    g, w0 = gamma, omega
 
    assert gamma <= w0 # gamma <= w0に限定
    w = np.sqrt(w0**2-g**2)
    phi = np.arctan(-g/w)
    A = 1/(2*np.cos(phi))
    
    x  = np.exp(-g*t)*2*A*np.cos(phi+w*t)
    
    return x


def FDM_dumper_sim(init_x, init_v, init_t, gamma, omega ,dt, T):
    '''
    FDM(finite-difference-method)によるバネ・マス・ダンパー系の減衰振動を解く関数
    
    Input for Function : 
    init_x  || 初期位相
    init_v  || 初期速度
    init_t  || 初期時刻
    gamma   || ダンパーの減衰係数
    omega   || 固有角周波数
    dt      || 時間ステップ
    T       || 合計シミュレーション時間
    
    Return values from Function :
    ts           || Time Steps 
    FDM_x        || 有限差分法による位相解
    FDM_v        || 有限差分法による速度解
    Analytical_x || 解析解
    diff         || 解析解との差分
    '''
    # initialize variables
    x = init_x 
    v = init_v 
    t = init_t 

    g, w0 = gamma, omega
    num_iter = int(T/dt)

    alpha = np.arctan(-1*g/np.sqrt(w0**2 - g**2))
    a = np.sqrt(w0**2 * x**2 / (w0**2 - g**2))

    # Initialize result data array
    ts = []
    FDM_x = []
    FDM_v = []
    Analytical_x = []
    diff = []

    # time step calculation loop
    for i in range(num_iter):
        fx = v
        fv = -1*w0**2 * x - 2*g * v
        
        #Update x/v/t
        x = x + dt * fx
        v = v + dt * fv
        t = t + dt
        x_a = a * np.exp(-1*g * t) * np.cos(np.sqrt(w0**2 - g**2) * t + alpha)
        d = x_a - x

        ts.append(t)
        FDM_x.append(x)
        Analytical_x.append(x_a)
        FDM_v.append(v)
        diff.append(d)

    return ts, FDM_x, FDM_v, Analytical_x, diff