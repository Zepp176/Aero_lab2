import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#                    ==== 1. Calibration ====

# Voltage at rest
V_zero_L = -0.199
V_zero_D = 0.666

# Calibration data
D_c = np.array([0, 0.06816, 0.181558, 0.294956, 0.408354, 0.521752, 0.63515, 0.748548, 0.975344])*9.81
V_D_c = np.array([0.65, 0.97, 1.51, 2, 2.57, 3.1, 3.62, 4.14, 5.12]).reshape((-1, 1))
L_c = np.array([0, 0.06816, 0.181558, 0.294956, 0.408354, 0.521752, 0.63515, 0.748548, 0.861946, 0.975344, 1.428936, 2.33612])*9.81
V_L_c = np.array([-0.21, -0.17, -0.12, -0.06, 0, 0.06, 0.12, 0.17, 0.22, 0.31, 0.55, 1.1]).reshape((-1, 1))

# Linear regressions to find sensitivity
reg_D = LinearRegression().fit(V_D_c, D_c)
a_D = reg_D.coef_[0]; b_D_c = reg_D.intercept_
reg_L = LinearRegression().fit(V_L_c, L_c)
a_L = reg_L.coef_[0]; b_L_c = reg_L.intercept_
b_D = -V_zero_D*a_D
b_L = -V_zero_L*a_L

V_D = np.array([0.5, 5.2])
V_L = np.array([-0.25, 1.2])

plt.figure(figsize=(7,5))

plt.plot(V_D, a_D*V_D + b_D_c, "g--")
plt.plot(V_L, a_L*V_L + b_L_c, "g--", label="_nolegend_")
plt.plot(V_D_c, D_c, "o")
plt.plot(V_L_c, L_c, "o")

plt.title("Calibration")
plt.grid()
plt.legend(["Linear regressions", "Drag calibration data", "Lift calibration data"])
plt.ylabel("Force [N]")
plt.xlabel("Tension [V]")
plt.tight_layout()
plt.savefig("fig1.png", dpi=300)

#                    ==== 2. Data ====

C_Darms = 0.06433
P = 300
angles = np.linspace(-6, 20, 14)
L = b_L + a_L*np.array([-0.373, -0.34, -0.306, -0.28, -0.24, -0.2, -0.135, -0.095, -0.05, -0.025, -0.065, -0.091, -0.094, -0.095])
D = b_D + a_D*np.array([0.96, 0.93, 0.9, 0.88, 0.86, 0.86, 0.87, 0.885, 0.92, 0.995, 1.178, 1.248, 1.32, 1.41])
rho = 1.225
S = 0.05*0.3
AR = 0.3/0.05

# upstream flow velocity
U = np.sqrt(2*P/rho)

# Lift and drag coefficients
C_L = L / (1/2 * rho * U**2 * S)
C_D = D / (1/2 * rho * U**2 * S) - C_Darms # Don't forget to substract C_D of the arms

# Reynolds number
Re = U*0.05/1.48e-5
print("Re: {:.0f}".format(Re))

# Regression of the C_L linear part
reg_CL = LinearRegression().fit(angles[:9].reshape((-1, 1)), C_L[:9])
a_CL = reg_CL.coef_[0]; b_CL = reg_CL.intercept_
offset_angle = b_CL/a_CL # We add an offset to enforce C_L = 0 when aoa = 0
angles += offset_angle

angles_CL = np.array([angles[0], angles[8]])
CL_reg = a_CL*angles_CL

plt.figure(figsize=(7,5))

plt.plot(angles, C_D, 'o-')
plt.plot(angles, C_L, 'o-')
plt.plot(angles_CL, CL_reg, 'k--')

plt.grid()
plt.ylabel("Aerodynamic coefficients")
plt.xlabel("angle [??]")
plt.legend(["$C_D$", "$C_L$", "$C_L$ regression slope = {:.2f}".format(a_CL*180/np.pi)])
plt.tight_layout()
plt.savefig("fig2.png", dpi=300)

#                    ==== 3. Interpolation of polar curve ====

# Ici on fait une regression avec les moindres carr??s pour fit les donn??es sur y = a + b x^2
#
#     ( 1   x_1**2   )
#     ( 1   x_2**2   )
# A = ( .    .       )
#     ( 1 x_(n-1)**2 )
#     ( 1   x_n**2   )
#
# x = [a b].T
#
# y = [y_1 y_2 ... y_(n-1) y_n].T
#
# Mais A n'est pas carr?? donc on r??sout:
#
# A x = b
# A.T A x = A.T b
# x = (A.T A)^-1 A.T b
#
# Ensuite on a, a = C_D0 et b = k

A = np.ones((9, 2))
for i in range(9):
    A[i,1] = C_L[i]**2
coefs = np.linalg.inv(A.T@A)@A.T@C_D[:9]

C_L_reg = np.linspace(-0.75, 0.8, 100)
C_D0 = coefs[0]
k = coefs[1]

C_D_reg = C_D0 + k*C_L_reg**2

# Find the maximum L/D with k and C_D0
max_LD = 1/np.sqrt(4*C_D0*k)

#                    ==== 4. Plot of the polar curve ====

plt.figure(figsize=(7,5))

plt.plot(C_D, C_L, 'o-')
plt.plot(C_D_reg, C_L_reg, 'k--')
plt.plot([0, 0.8/max_LD], [0, 0.8])
plt.plot([1/(max_LD**2*2*k)], [1/(max_LD*2*k)], 'xr', markersize=10)

# Lift at maximum L/D
L_at_max_LD = 1/(max_LD*2*k)
# Angle of attack at maximum L/D
max_angle = L_at_max_LD/a_CL

plt.title("maximum L/D: {:.2f} @ {:.2f}?? aoa".format(max_LD, max_angle))
plt.legend(["Polar curve", "Parabolic interpolation", "Maximum slope", "Optimal aerodynamics"])
plt.grid()
plt.xlabel("$C_D$")
plt.ylabel("$C_L$")
plt.tight_layout()
plt.savefig("fig3.png", dpi=300)

# Oswald efficiency
e = 1/(np.pi*k*AR)
print("Oswald efficiency: {:.3f}".format(e))

