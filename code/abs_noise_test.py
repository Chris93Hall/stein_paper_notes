"""
abs_noise_test.py
"""

import numpy as np

def test():
    nlen = 100000
    s1_list = list(np.arange(.1, 10))
    s2_list = list(np.arange(.1, 10))
    n1_var_list = list(np.arange(.1, 10, 1))
    n2_var_list = list(np.arange(.1, 10, 1))
    s1_list = [float(xx) for xx in s1_list]
    s2_list = [float(xx) for xx in s2_list]
    n1_var_list = [float(xx) for xx in n1_var_list]
    n2_var_list = [float(xx) for xx in n2_var_list]
    res_var_list = []
    pred_mat = []
    for s1 in s1_list:
        print(f's1: {s1}')
        for s2 in s2_list:
            print(f's2: {s2}')
            for var1 in n1_var_list:
                for var2 in n2_var_list:
                    res_var = do_trial(s1, s2, var1, var2, nlen)
                    res_var_list.append(res_var)
                    pred_mat.append([s1, s2, var1, var2,
                                     s1**2, s2**2, var1**2, var2**2,
                                     s1**3, s2**3, var1**3, var2**3,
                                     s1**4, s2**4, var1**4, var2**4,
                                     s1**(-1), s2**(-1), var1**(-1), var2**(-1),
                                     var1*var2, s1*var2, s2*var1, var1*s2*s2, var2*s1*s1,
                                     var1*s1, var2*s2, var1*s1*s1, var2*s2*s2, var1*var2*s1*s2,
                                     s1**(-2), s2**(-2), var1**(-2), var2**(-2),
                                     s1**(-4), s2**(-4), var1**(-4), var2**(-4)])

    coeffs = predict_fit(pred_mat, res_var_list)
    print(coeffs)
    input('continue...')

    # verify fit
    s1_list = list(np.arange(.1, 5, 0.1))
    s2_list = list(np.arange(.1, 5, 0.1))
    n1_var_list = list(np.arange(.1, 5, 0.1))
    n2_var_list = list(np.arange(.1, 5, 0.1))
    s1_list = [float(xx) for xx in s1_list]
    s2_list = [float(xx) for xx in s2_list]
    n1_var_list = [float(xx) for xx in n1_var_list]
    n2_var_list = [float(xx) for xx in n2_var_list]
    for s1 in s1_list:
        for s2 in s2_list:
            for var1 in n1_var_list:
                for var2 in n2_var_list:
                    res_actual = do_trial(s1, s2, var1, var2, nlen)
                    res_pred = np.sum([s1*coeffs[0], s2*coeffs[1], var1*coeffs[2], var2*coeffs[3],
                    coeffs[4]*s1**2, coeffs[5]*s2**2, coeffs[6]*var1**2, coeffs[7]*var2**2,
                    coeffs[8]*s1**3,  coeffs[9]*s2**3, coeffs[10]*var1**3, coeffs[11]*var2**3,
                    coeffs[12]*s1**4,  coeffs[13]*s2**4, coeffs[14]*var1**4, coeffs[15]*var2**4,
                    coeffs[16]*s1**(-1), coeffs[17]*s2**(-1), coeffs[18]*var1**(-1), coeffs[19]*var2**(-1),
                    coeffs[20]*var1*var2, coeffs[21]*s1*var2, coeffs[22]*s2*var1,
                    coeffs[23]*var1*s2*s2, coeffs[24]*var2*s1*s1,
                    coeffs[25]*var1*s1, coeffs[26]*var2*s2, coeffs[27]*var1*s1*s1, coeffs[28]*var2*s2*s2, coeffs[29]*var1*var2*s1*s2,
                    coeffs[30]*s1**(-2), coeffs[31]*s2**(-2), coeffs[32]*var1**(-2), coeffs[33]*var2**(-2),
                    coeffs[34]*s1**(-4), coeffs[35]*s2**(-4), coeffs[36]*var1**(-4), coeffs[37]*var2**(-4)])
                    print(f'{res_actual}: {res_pred}')
    return

def do_trial(s1, s2, var1, var2, nlen):
    vec1 = np.random.normal(scale=np.sqrt(var1/2.0), size=nlen).astype(complex)*1j + np.random.normal(scale=np.sqrt(var1/2.0), size=nlen)
    vec2 = np.random.normal(scale=np.sqrt(var2/2.0), size=nlen).astype(complex)*1j + np.random.normal(scale=np.sqrt(var2/2.0), size=nlen)
    res = np.abs(s1*vec2 + s2*vec1 + vec1*vec2)
    return np.var(res)

def predict_fit(pred_mat, res_vec):
    pred_mat = np.array(pred_mat)
    res_vec = np.array(res_vec)
    res_vec = res_vec[:, np.newaxis]
    estimator = np.matmul(np.linalg.inv(np.matmul(pred_mat.T, pred_mat)), pred_mat.T)
    print(np.shape(estimator))
    print(np.shape(res_vec))
    coeffs = np.matmul(estimator, res_vec)
    return coeffs.squeeze().tolist()

if __name__ == '__main__':
    test()
