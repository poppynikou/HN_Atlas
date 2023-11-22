import numpy as np 

a = 0
b = 1
n = 100
h = (b - a) / (n - 1)
x = np.linspace(a, b, n)

coef1 =  0.69063112
coef2 = -0.82832175
coef3 = -0.03050362

f = np.sqrt(1+ (3*coef1*x**2 + 2*coef2*x + coef3)**2)

I_simp = (h/3) * (f[0] + 2*sum(f[:n-2:2]) + 4*sum(f[1:n-1:2]) + f[n-1])

print(I_simp)

import matplotlib.pyplot as plt 
plt.scatter(x,(coef1*x**3 + coef2*x**2 + coef3*x))
plt.show()