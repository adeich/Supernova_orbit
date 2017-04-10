from __future__ import division
import numpy as np
from time import time
from scipy.integrate import odeint


def get_SI(name):
	phys_constants = {
	'aEarthSun': 1.5 * 10**11, # meters
	'eEarth': 0.06, 
	'mEarth': 6. * 10**24, # kg
	'mSun': 1.99 * 10**30, # kg  
	}

	return phys_constants[name]

# returns [circum_steps] x 6 array. 012 are 3position (z is 0); 345 are 3velocity
# (vz is 0).
def generate_orbit_pos_vel(a, e, m1, m2, circum_steps,
		const_time_or_angle=None, debug=True):
	M = m1; m = m2
	G = 6.67 * 10**-11
	mu = (m * M) / (m + M)           # reduced mass. 
	E = -1 * ((G * M * m) / (2 * a)) # total energy.
	T = np.sqrt((4 * (np.pi**2) * a**3)/(G * (M + m))) # orbital period.
	L = np.sqrt(G * mu * a * (1 - e**2)) # angular momentum per unit mass
	dt = T / circum_steps
	q = M / float(m) # useful ratio; see below.
	Gm = G * (m1 + m2)
	r0_vec = ((1 - e)/(1 + q)) * np.array([a, 0, 0]) # initial position. 
	v0_mag = (1 + q)**(-1) * ((1 + e)/(1 - e))**(0.5) * ((G * (m + M))/(a)) 
	v0_vec = np.array([0, v0_mag, 0])
	
	# orbit integration function for scipy.integrate.odeint.
	def dy_dt(y_now, t):
		y_next = np.zeros(6)  #  x, y, z, vx, vy, vz
		y_next[:3] = y_now[3:]    #  dx/dt = v
		one_over_r_cubed = ((y_next[:3]**2).sum())**-1.5
		y_next[3:] = - Gm * y_now[:3] * one_over_r_cubed
		return y_next

	t_array = np.linspace(0, T, circum_steps)
	y0 = np.concatenate([r0_vec, v0_vec]) 

	start_time = time()
	y_out = odeint(dy_dt, y0, t_array, full_output=debug)
	if debug:
		print("{} steps took {} seconds".format(circum_steps, time() - start_time))
		
	return y_out

	

# Given a body at radius vector (x, y, z) and velocity vector (vx, vy, 0),
# this function computes and returns a tuple (a, e) 
# representing the two-body semi-major axis and eccentricity.
# Vectors may be of either 2 or 3 dimensions.
def NewAE(m, M, r_vec, v_vec):
  mu = (m * M) / (m + M)                        # reduced mass.
  E = - ((G * m * M) / np.linalg.norm(r_vec) + (0.5) *  # energy. 
    mu * np.linalg.linalg.dot(v_vec, v_vec))
  a = - (G * m * M) / (2 * E)        # semi-major axis.
  J = mu * np.cross(r_vec, v_vec)    # angular momentum.
  e = np.sqrt(1 - (np.linalg.norm(J)**2)/(mu**2 * (m + M) * G * a)) # eccentricity
  return a, e

