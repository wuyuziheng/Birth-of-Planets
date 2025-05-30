import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np

G = 6.674e-11

def ellipse_orbit(theta, v, phi, r, sun_mass):
    h = v*math.sin(phi)*r
    E = (1/2)*v*v - (G * sun_mass)/r
    e = math.sqrt(1+(2*E*h*h)/(G*G*sun_mass*sun_mass))
    return (h**2/(G*sun_mass))/(1+e*np.cos(theta))

def draw_polar(series: pd.Series):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')  
    ax.set_theta_direction(-1) 
    ax.set_theta_zero_location('N')  
    ax.plot(series.index, series.values, '--')
    plt.savefig("./orbit.png")
    plt.close(fig)

config = {"v": 10, "phi": math.pi/2, "r": 10000, "sun_mass": 1.0e16}
theta = np.arange(0, 2*math.pi, 0.01)
orbit = pd.Series(ellipse_orbit(theta, **config), theta)
draw_polar(orbit)