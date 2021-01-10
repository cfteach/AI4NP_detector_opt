#-------- credits:
# detector toy model inspired to example from Yandex, yandex.com
# readapted and expanded for AI4NP --- C.F. 

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


y_min =-10.1
y_max = 10.1

class Tracker(object):

    def __init__(self, R, pitch, y1, y2, y3, z1, z2, z3):
        """
        Generates Z, Y coordinates of straw tubes of the tracking system.

        Parameters:
        -----------
        R : float
            Radius of a straw tube.
        pitch : float
            Distance between two adjacent tubes in one layer of the system.
        y1 : float
            Shift between two layers of tubes.
        y2 : float
            Shift between two layers of tubes.
        y3 : float
            Shift between two layers of tubes.
        z1 : float
            Shift between two layers of tubes.
        z2 : float
            Shift between two layers of tubes.
        z3 : float
            Shift between two layers of tubes.
        """

        self.R = R
        self.pitch = pitch
        self.y1 = y1
        self.y2 = y2
        self.y3 = y3
        self.z1 = z1
        self.z2 = z2
        self.z3 = z3


    def create_geometry(self):
        """
        Generate Z, Y coordinates of the tubes.
        """

        base = np.arange(-100, 101, 1)
        step = self.pitch

        layer1_y = step * base

        layer1_z = 0. * np.ones(len(base))

        layer2_y = layer1_y + self.y1
        layer2_z = layer1_z + self.z1

        layer3_y = layer2_y + self.y2
        layer3_z = layer2_z + self.z2

        layer4_y = layer3_y + self.y3
        layer4_z = layer3_z + self.z3

        Z = np.concatenate((layer1_z.reshape(-1, 1),
                            layer2_z.reshape(-1, 1),
                            layer3_z.reshape(-1, 1),
                            layer4_z.reshape(-1, 1)), axis=1)

        Y = np.concatenate((layer1_y.reshape(-1, 1),
                            layer2_y.reshape(-1, 1),
                            layer3_y.reshape(-1, 1),
                            layer4_y.reshape(-1, 1)), axis=1)

        geo = [Z, Y]

        return geo


def geometry_display(Z, Y, R, y_min=-10, y_max=10, block=True, pause =5):
    """
    Displays straw tubes of the tracking system.

    Parameters:
    -----------
    Z : array_like
        Array of z-coordinates of the tubes.
    Y : array_like
        Array of y-coordinates of the tubes.
    R : float
        Radius of a tube.
    y_min : float
        Minimum y-coordinate to display.
    y_max : float
        Maximum y-coordinate to display.
    """

    Z_flat = np.ravel(Z)
    Y_flat = np.ravel(Y)

    z_min = Z_flat.min()
    z_max = Z_flat.max()

    sel = (Y_flat >= y_min) * (Y_flat < y_max)
    Z_flat = Z_flat[sel]
    Y_flat = Y_flat[sel]

    plt.figure(figsize=(6, 6 * (y_max - y_min + 2) / (z_max - z_min + 10)))
    plt.scatter(Z_flat, Y_flat)


    for z,y in zip(Z_flat, Y_flat):
        circle = plt.Circle((z, y), R, color='b', fill=False)
        plt.gcf().gca().add_artist(circle)

    #print("........len(Z_flat): ", len(Z_flat))

    plt.xlim(z_min - 5, z_max + 5)
    plt.ylim(y_min - 1, y_max + 1)
    plt.xlabel('Z', size=14)
    plt.ylabel('Y', size=14)
    plt.xticks(size=14)
    plt.yticks(size=14)

    #fig = plt.figure()
    #timer = fig.canvas.new_timer(interval = 3000) #creating a timer object and setting an interval of 3000 milliseconds
    #timer.add_callback(close_event)
    #timer.start()
    #plt.show()

    """
    #plt.show(block=block) #block python execution
    if(block==False):
      if(pause==-1):
        #plt.figure(0)
        #plt.ion()
        #plt.show()
        pass 
      else:
        #plt.pause(pause)
        #plt.close()
        pass

    """

    #num_wires = len(Y_flat)
    #return num_wires








class Tracks(object):

    def __init__(self, b_min, b_max, alpha_mean, alpha_std):
        """
        Generates tracks.

        Parameters:
        -----------
        b_min : float
            Minimum y intercept of tracks.
        b_max : float
            Maximum y intercept of tracks.
        alpha_mean : float
            Mean value of track slopes.
        alpha_std : float
            Standard deviation of track slopes.
        """

        self.b_min = b_min
        self.b_max = b_max
        self.alpha_mean = alpha_mean
        self.alpha_std = alpha_std

    def generate(self, N):
        """
        Generates tracks.

        Parameters:
        -----------
        N : int
            Number of tracks to generate.

        Returns:
        --------
        tracks : array-like
            List of track parameters [[k1, b1], [k2, b2], ...]
        """

        B = np.random.RandomState(42).uniform(self.b_min, self.b_max, N)
        Angles = np.random.RandomState(42).normal(self.alpha_mean, self.alpha_std, N)
        K = np.tan(Angles)

        tracks = np.concatenate((K.reshape(-1, 1), B.reshape(-1, 1)), axis=1)

        return tracks


def tracks_display(tracks, Z, block=True, pause=5):
    """
    Displays tracks.

    Parameters:
    -----------
    tracks : array-like
        List of track parameters.
    Z : array-like
        List of z-coordinates.
    """

    Z_flat = np.ravel(Z)

    z_min = Z_flat.min()
    z_max = Z_flat.max()

    z1 = z_min - 5
    z2 = z_max + 5

    for k, b in tracks:

        plt.plot([z1, z2], [k * z1 + b, k * z2 + b], c='0.2', alpha=0.3)

    #fig = plt.figure()
    #timer = fig.canvas.new_timer(interval = 3000) #creating a timer object and setting an interval of 3000 milliseconds
    #timer.add_callback(close_event)
    #timer.start()
    #plt.show()

    #plt.show()

    
    plt.show(block=block)

    if(block==False):
      if(pause==-1):
        #plt.figure(0)
        #plt.ion()
        plt.show()
      else:
        plt.pause(pause)
        plt.close()
        #plt.show()
    


def get_score(Z, Y, tracks, R):
    """
    Score of the tracking system geometry.
    Z : array_like
        Array of z-coordinates of the tubes.
    Y : array_like
        Array of y-coordinates of the tubes.
    R : float
        Radius of a tube.
    tracks : array-like
        List of track parameters.
    """

    values = []

    reso = []
    values_reso = []

    for k, b in tracks:

        Y_pred = k * Z + b
        dY = np.abs(Y_pred - Y)

        alpha = np.arctan(k)
        cos = np.cos(alpha)

        #is_intersect = dY * cos < R
        is_intersect = dY < R
        n_intersections = (is_intersect).sum()

        masked_dY = dY[np.array(is_intersect)]


        if n_intersections >= 2:
            values.append(1)

            tmp_reso = np.mean(masked_dY)
            values_reso.append(tmp_reso)
        else:
            values.append(0)


    #--------- looking at all tracks ---------#

    fin_reso = 99999  #fictitious value for resolution

    #print("len(values_reso): ", len(values_reso))

    if(len(values_reso)>0):
        fin_reso = np.mean(values_reso)
        # fictitious value for resolution

    # N.B. the residual from the wire centre is a proxy of the resolution for this toy-model
    #print("final reso: ", fin_reso)

    return np.mean(values),fin_reso


def plot_objective(min_objective_values):
    """
    Plot optimization curve
    """
    plt.figure(figsize=(9, 6))
    plt.plot(min_objective_values, linewidth=2)
    plt.xlabel("Number of calls", size=14)
    plt.ylabel('Objective', size=14)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.title('Optimization curve', loc='right', size=14)
    plt.grid(b=1)


def calculate_wires(Y, y_min, y_max):

  Y_flat_ = np.ravel(Y)
  sel = (Y_flat_ >= y_min) * (Y_flat_ < y_max)
  Y_flat_ = Y_flat_[sel]

  num_wires = len(Y_flat_)
  return num_wires

def wires_volume(Y, y_min, y_max,R):

  num_wires = calculate_wires(Y, y_min, y_max)
  vol = num_wires*4.*3.14 * (R**2)
  return vol


def close_event():
    plt.close() #timer calls this function after 3 seconds and closes the window
