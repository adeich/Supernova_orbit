from __future__ import division
import numpy as np
from time import time
from scipy.integrate import odeint
import math
import numpy


def get_SI(name):
	phys_constants = {
	'aEarthSun': 1.5 * 10**11, # meters
	'eEarth': 0.06, 
	'mEarth': 5.97 * 10**24, # kg
	'mSun': 1.99 * 10**30, # kg	
	'G': 6.67 * 10**-11
	}

	return phys_constants[name]

# returns [circum_steps] x 6 array. 012 are 3position (z is 0); 345 are 3velocity
# (vz is 0).
def generate_orbit_pos_vel(a, e, m1, m2, circum_steps,
		const_time_or_angle=None, debug=True):
	M = m1; m = m2
	G = 6.67 * 10**-11
	mu = (m * M) / (m + M)					 # reduced mass. 
	E = -1 * ((G * M * m) / (2 * a)) # total energy.
	T = np.sqrt((4 * (np.pi**2) * a**3)/(G * (M + m))) # orbital period.
	L = np.sqrt(G * mu * a * (1 - e**2)) # angular momentum per unit mass
	q = m1 / m2 # useful ratio; see below.

	# start at r_max and therefore v_min.
	r0_vec = a * (1 - e) * np.array([1, 0, 0]) # initial position. 
	K = -G * m1 * m2
	v0_mag = ((-K/(mu * a)) * (1-e)/(1+e))**(0.5)
	v0_vec = np.array([0, v0_mag, 0])
	
	# orbit integration function for scipy.integrate.odeint.
	def dy_dt(y, t):
		r = np.sqrt(y[0]**2 + y[1]**2 + y[2]**2)
		dy0 = y[3]; dy1 = y[4]; dy2 = y[5]
		dy3 = -(mu / (r**3)) * y[0]
		dy4 = -(mu / (r**3)) * y[1]
		dy5 = -(mu / (r**3)) * y[2]
		return [dy0, dy1, dy2, dy3, dy4, dy5]

	t_array = np.linspace(0, T, circum_steps)
	y0 = np.concatenate([r0_vec, v0_vec]) 

	start_time = time()
	y_out = odeint(dy_dt, y0, t_array, full_output=False)
	if debug:
		print("{} steps took {} seconds".format(circum_steps, time() - start_time))

	if debug: 
		debug_dict = {'M': M,
		'E': E, 'T': T, 'r0_vec': r0_vec, 'v0_vec': v0_vec
		}
		for key, item in debug_dict.iteritems():
			print("{}: {}".format(key, item))
		
	return y_out

	

def generate_orbit_pos_vel2(a, e, m1, m2, circum_steps):
	G = 6.67 * 10**-11 
	m = m1;	M = m2

	# reduced mass
	mu = (m * M) / (m + M)

	# total energy
	E = -1 * ((G * M * m) / (2 * a))

	# orbital period
	period = math.sqrt((4 * (math.pi**2) * a**3)/(G * (M + m)))

	# angular momentum per unit mass
	L = math.sqrt(G * M * a * (1 - e**2))

	# number of discrete orbit locations, as a function of e.
	steps = circum_steps #math.floor(1 / (0.07 * (1 - e)))

	# timestep
	dt =  1000*period / steps

	output = np.zeros([circum_steps, 7])

	# initialize incremental step parameters
	step = 0 
	t = 0 
	theta = 0 
	dtheta = None
	
	# step around the orbital path once.
	while step < steps:
		r = (a * (1 - e**2))/(1 + e * math.cos(theta))
		x = r * math.cos(theta)
		y = r * math.sin(theta)
		dx = (a * e * (1 - e**2) * math.cos(theta) * math.sin(theta))/((1 + e * math.cos(theta))**2) - (a * (1 - e**2)) * math.sin(theta) / (1 + e * math.cos(theta))
		dy = (a * e * (1 - e**2) * math.cos(theta))/(1 + e * math.cos(theta)) + (a * e * (1 - e**2) * (math.sin(theta))**2) / (1 + e * (math.cos(theta))**2)
		V = math.sqrt((2/mu) * ((G * m * M) / r + E)) 
		vx = V/math.sqrt(dx**2 + dy**2) * dx
		vy = V/math.sqrt(dx**2 + dy**2) * dy

		# put values into 3d vectors (with z value 0)
		aPosition = numpy.array([float(x), float(y), 0]) 
		aVelocity = numpy.array([float(vx), float(vy), 0]) 
		output[step] = numpy.concatenate([aPosition, aVelocity, np.array([theta])])
		step += 1
		if (L/(r**2)) * dt < 2:
			dtheta = (L/r**2) * dt
		else:
			dtheta = 2 
		theta += dtheta
		t += dt

	return output



# Given a body at radius vector (x, y, z) and velocity vector (vx, vy, 0),
# this function computes and returns a tuple (a, e) 
# representing the two-body semi-major axis and eccentricity.
# Vectors may be of either 2 or 3 dimensions.
def NewAE(m, M, r_vec, v_vec):
	mu = (m * M) / (m + M)												# reduced mass.
	E = - ((G * m * M) / np.linalg.norm(r_vec) + (0.5) *	# energy. 
		mu * np.linalg.linalg.dot(v_vec, v_vec))
	a = - (G * m * M) / (2 * E)				# semi-major axis.
	J = mu * np.cross(r_vec, v_vec)		# angular momentum.
	e = np.sqrt(1 - (np.linalg.norm(J)**2)/(mu**2 * (m + M) * G * a)) # eccentricity
	return a, e

