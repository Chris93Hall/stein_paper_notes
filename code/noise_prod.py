"""
noise_prod.py
"""

import numpy as np


def test1(var1 = 1.0, var2=1.0):
    #var1 = 1.0
    #var2 = 1.0
    nlen = 1000000
    vec1 = np.random.normal(scale=np.sqrt(var1/2.0), size=nlen).astype(complex)*1j + np.random.normal(scale=np.sqrt(var1/2.0), size=nlen)
    vec2 = np.random.normal(scale=np.sqrt(var2/2.0), size=nlen).astype(complex)*1j + np.random.normal(scale=np.sqrt(var2/2.0), size=nlen)
    res = np.abs(vec1*vec2)
    res2 = vec1*vec2
    print(f'var1: {var1}')
    print(f'var2: {var2}')
    print('|n1 * n2|')
    print(f'res var: {np.var(res)}')
    prediction = var1*var2*0.38
    print(f'predicted var: {prediction}')
    print('n1 * n2')
    prediction = var1*var2
    print(f'res2 var: {np.var(res2)}')
    print(f'predicted var: {prediction}')

def test2(var1=1.0, var2=1.0, var3=1.0, var4=1.0):
    nlen = 10000
    vec1 = np.random.normal(scale=np.sqrt(var1/2.0), size=nlen).astype(complex)*1j + np.random.normal(scale=np.sqrt(var1/2.0), size=nlen)
    vec2 = np.random.normal(scale=np.sqrt(var2/2.0), size=nlen).astype(complex)*1j + np.random.normal(scale=np.sqrt(var2/2.0), size=nlen)
    vec3 = np.random.normal(scale=np.sqrt(var3/2.0), size=nlen).astype(complex)*1j + np.random.normal(scale=np.sqrt(var3/2.0), size=nlen)
    vec4 = np.random.normal(scale=np.sqrt(var4/2.0), size=nlen).astype(complex)*1j + np.random.normal(scale=np.sqrt(var4/2.0), size=nlen)
    res = np.abs(vec1 + vec2 + (vec3*vec4)) 
    print(f'var1: {var1}')
    print(f'var2: {var2}')
    print(f'var3: {var3}')
    print(f'var4: {var4}')
    print(f'res var: {np.var(res)}')


def test3(var1=1.0, var2=1.0, scale1=1.0, scale2=1.0):
    nlen = 1000000
    vec1 = np.random.normal(scale=np.sqrt(var1/2.0), size=nlen).astype(complex)*1j + np.random.normal(scale=np.sqrt(var1/2.0), size=nlen)
    vec2 = np.random.normal(scale=np.sqrt(var2/2.0), size=nlen).astype(complex)*1j + np.random.normal(scale=np.sqrt(var2/2.0), size=nlen)
    res = np.abs((scale1*vec2) + (scale2*vec1) + (vec1*vec2)) 
    print(f'var1: {var1}')
    print(f'var2: {var2}')
    print(f'scale1: {scale1}')
    print(f'scale2: {scale2}')
    print(f'res var: {np.var(res)}')
    const1 = (4.0 - np.pi)/2.0
    upper_bound = (const1*((scale1**2)*var2 + (scale2**2)*var1)) + (0.38*var1*var2)
    print(f'res var upper bound: {upper_bound}')
    lower_bound = (const1*((scale1**2)*var2 + (scale2**2)*var1))
    print(f'res var lower bound: {lower_bound}')
    res2 = ((1.0/var1) + (1.0/var2) + (1.0/(var1*var2)))**(-1)
    print(f'traditional answer: {res2}')



