#!bin/python3

import time
import cramer
import gauss
from numpy import linalg, isnan
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def power(xdata, a, b):
    return list(map(lambda x: a * pow(x, b), xdata))


sizes = []

times_cramer = []
times_gauss = []
times_builtin = []

errors_cramer = []
errors_gauss = []
errors_builtin = []


NN = [10, 20, 30, 40, 50, 100, 200, 250, 400, 500, 600, 800, 1000]
for size in NN:
    examp = cramer.generator(size, 5, 2, 3)
    A, B = examp
    # ---------Cramer test---------
    t1 = time.perf_counter()
    X = cramer.cramer(A, B)
    t2 = time.perf_counter()
    error_cramer = cramer.mxw(A, X) - B
    errors_cramer.append(max(abs(error_cramer[0])))
    delta_cramer = t2 - t1
    print("\ndet matrix of size {} by cramer calculated in {}".format(A.shape[0], t2 - t1))
    times_cramer.append(delta_cramer)

    # ---------Gauss test---------
    t1 = time.perf_counter()
    X = gauss.gaussian_elimination(A, B)
    t2 = time.perf_counter()
    error_gauss = cramer.mxw(A, X) - B
    errors_gauss.append(max(abs(error_gauss[0])))
    delta_gauss = t2 - t1
    print("\ndet matrix of size {} by gauss calculated in {}".format(A.shape[0], t2 - t1))
    times_gauss.append(delta_gauss)

    # ---------BuiltIn test---------
    t1 = time.perf_counter()
    X = linalg.solve(A, B)
    t2 = time.perf_counter()
    error_builtin = cramer.mxw(A, X) - B
    errors_builtin.append(max(abs(error_builtin)))
    delta_builtin = t2 - t1
    print("\ndet matrix of size {} by builtin calculated in {}".format(A.shape[0], t2 - t1))
    times_builtin.append(delta_builtin)

    sizes.append(size)

# errors_cramer = [x for x in errors_cramer if str(x) != 'nan']
box = [errors_cramer, errors_gauss, errors_builtin]

print(box)
popt_cramer, pcov_cramer = curve_fit(power, sizes, times_cramer)
popt_gauss, pcov_gauss = curve_fit(power, sizes, times_gauss)
popt_builtin, pcov_builtin = curve_fit(power, sizes, times_builtin)

print(popt_cramer)
print(popt_gauss)
print(popt_builtin)

x = range(max(sizes))
plt.scatter(sizes, times_cramer, color='r', label='Cramer')
plt.scatter(sizes, times_gauss, color='g', label='Gauss')
plt.scatter(sizes, times_builtin, color='b', label='BuiltIn')
plt.plot(x, power(x, *popt_cramer), 'r--', label='fit: a=%5.3f*, b=%5.3f' % tuple(popt_cramer))
plt.plot(x, power(x, *popt_gauss), 'g--', label='fit: a=%5.3f*, b=%5.3f' % tuple(popt_gauss))
plt.plot(x, power(x, *popt_builtin), 'b--', label='fit: a=%5.3f*, b=%5.3f' % tuple(popt_builtin))
#plt.ylim(0, 15)
plt.legend(loc='upper left')
plt.title('Performance of algorithms')
plt.xlabel('size matrix [ ]')
plt.ylabel('CPU time [sec]')
plt.show()


plt.title('Errors of algorithms')
plt.boxplot(box)
plt.axhline(y=0.0, color='r', linestyle='-')
plt.xticks([1, 2, 3], ['Cramer', 'Gauss', 'BuiltIn'])
plt.show()