import numpy as np
import pareto_front as pf
import matplotlib.pyplot as plt

x = np.array([[1,2], [2, 3], [4, 1], [1, 1], [2,2], [3, 2.5]])

pfismin = pf.is_pareto_efficient_indexed(x, True)
print(x[pfismin])

pfismax = pf.is_pareto_efficient_indexed(x, False)
print(x[pfismax])

plt.figure()
plt.scatter(x[:,0], x[:,1])
plt.show()



