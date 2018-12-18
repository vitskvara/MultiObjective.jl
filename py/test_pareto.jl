using MultiObjective
using PyPlot

x = Array([1 2; 2 3; 4 1; 1 1; 2 2; 3 2.5]')

pfismin = MultiObjective.is_pareto_efficient(x, true)
print(x[:,pfismin])

pfismax = MultiObjective.is_pareto_efficient(x, false)
print(x[:,pfismax])

figure()
scatter(x[1,:], x[2,:])
show()
