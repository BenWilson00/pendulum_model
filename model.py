import math
import numpy
import scipy.integrate

G = 9.81     # m.s^-2;         gravitational field strength
M = 0.24     # kg;             mass of bob
CD = 2.00    #                 approx. drag coefficient
CSA = 0.016    # m^2             approx. CSA
P_AIR = 1.23 # kg.m^-3         approx. air density
R = 0.98     # m               pendulum length

k = CD*CSA*P_AIR/2          # coefficient of v^2 in drag force
L = 0.010                   # coefficient of v in simplified drag force (pulled out of the air)

# timesteps
n_timesteps = 9000
total_time = 45.0     # seconds
t_per_step = total_time / (n_timesteps - 1)

timesteps = [t_per_step*i for i in range(n_timesteps)]


data = {
  "thetas" : [],
  "omegas" : [],
  "alphas" : []
}

# Initial conditions
initials = [1, 0, 1, 0] #rad, rad.s^-1

def eq_system(variables, timesteps, params):
  theta, omega, t2, o2 = variables
  k, R, M, G = params

  return [
    omega,                                                                   # == theta'
    -(k*(R**2)*(omega**2)*numpy.sign(omega) + 0.05*omega + M*G*math.sin(theta))/(M*R),    # == omega',
    o2,                      # == t2'
    -(L*R*o2 + M*G*t2)/(M*R) # == o2'
  ]


solution = scipy.integrate.odeint(eq_system, initials, timesteps, args=((k, R, M, G),) )

# energy functions
GPE = lambda x: M*G*R*(1-math.cos(x))
GPE2 = lambda x: M*G*R*x**2/2
KE = lambda x: M*(x**2)*(R**2)/2


energy = [(GPE(theta)+ KE(omega), GPE2(t2) + KE(o2)) for theta, omega, t2, o2 in solution]
energy2 = [(GPE(theta), KE(omega)) for theta, omega, t2, o2 in solution]

sol_with_times = zip(timesteps, *solution)

efficiency = [(energy[i][0] - energy[i-1][0])/(t_per_step*energy[i-1][0]) for i in range(1, len(solution))]
efficiency2 = [(energy[i][1] - energy[i-1][1])/(t_per_step*energy[i-1][1]) for i in range(1, len(solution))]

start = 0#int(5/t_per_step)



import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Plot 1

p3 = mpatches.Patch(color='dodgerblue',       label='Theta (rad), Small Angle')
p4 = mpatches.Patch(color='limegreen',  label='Omega (rad/s), Small Angle')
p1 = mpatches.Patch(color='r',          label='Theta (rad), Non-linear')
p2 = mpatches.Patch(color='orange',     label='Omega (rad/s), Non-linear')
plt.legend(handles=[p1, p2, p3, p4])

plt.ylabel("Angle (rad)  |  Angular velocity (rad/s)")
plt.xlabel("Time (s)")
plt.title("Angle of Displacement & Angular Velocity versus Time")


plt.plot(timesteps[start:], [i[2] for i in solution[start:]], color="dodgerblue", linestyle="--")
plt.plot(timesteps[start:], [i[3] for i in solution[start:]], color="limegreen", linestyle="--")
plt.plot(timesteps[start:], [i[0] for i in solution[start:]], color="r")
plt.plot(timesteps[start:], [i[1] for i in solution[start:]], color="orange")

plt.show()

# Plot 2

p1 = mpatches.Patch(color='r',         label='Energy, Non-linear')
p2 = mpatches.Patch(color='limegreen', label='Energy, Small Angle')
plt.legend(handles=[p1, p2])

plt.ylabel("Energy (J)")
plt.xlabel("Time (s)")
plt.title("Energy versus Time")

plt.plot(timesteps[start:], [i[0] for i in energy[start:]], color="r")
plt.plot(timesteps[start:], [i[1] for i in energy[start:]], color="limegreen", linestyle="--")

plt.show()

# Plot 3

p1 = mpatches.Patch(color='r',          label='GPE')
p2 = mpatches.Patch(color='orange',     label='KE')
p3 = mpatches.Patch(color='limegreen',  label='Total')
plt.legend(handles=[p1, p2, p3])

plt.ylabel("Energy (J)")
plt.xlabel("Time (s)")
plt.title("Energy versus Time (by Component, Non-linear)")

plt.plot(timesteps[start:], [i[0] for i in energy2[start:]], color="r")
plt.plot(timesteps[start:], [i[1] for i in energy2[start:]], color="orange")
plt.plot(timesteps[start:], [i[0] for i in energy[start:]], color="limegreen", linestyle="--")

plt.show()

# Plot 4

plt.ylabel("Efficiency (/s)")
plt.xlabel("Time (s)")
plt.title("Efficiency versus Time (Non-linear)")

reg_fit_coeffs = numpy.polyfit(timesteps[start+1:], efficiency[start:], 2)
reg_fit = numpy.poly1d(reg_fit_coeffs)

p1 = mpatches.Patch(color='orange',          label='Efficiency')
p2 = mpatches.Patch(color='limegreen',  label='Regression fit')
plt.legend(handles=[p1, p2])
plt.plot(timesteps[start+1:6000], efficiency[start:5999], color="orange", linestyle="--")
plt.plot(timesteps[start+1:6000], reg_fit(timesteps[start+1:6000]), color="limegreen")

plt.show()

# Plot 5

plt.ylabel("Efficiency (/s)")
plt.xlabel("Time (s)")
plt.title("Efficiency versus Time (Small Angle)")

reg_fit_coeffs = numpy.polyfit(timesteps[start+1:], efficiency2[start:], 1)
reg_fit = numpy.poly1d(reg_fit_coeffs)

p1 = mpatches.Patch(color='orange',          label='Efficiency')
p2 = mpatches.Patch(color='limegreen',  label='Regression fit')
plt.legend(handles=[p1, p2])
plt.plot(timesteps[start+1:], efficiency2[start:], color="orange", linestyle="--")
plt.plot(timesteps[start+1:], reg_fit(timesteps[start+1:]), color="limegreen")

plt.show()

#with open("output.txt", "w") as f:
#  f.write("\n".join( [", ".join(map(str, [i[0], i[1][0], i[1][1]])) for i in zip(timesteps, energy)] ))
# f.close()