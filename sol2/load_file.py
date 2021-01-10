import pickle
import dill
import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.model.problem import Problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.factory import get_visualization, get_decomposition
from pymoo.util.display import Display

from pymoo.factory import get_performance_indicator
from pymoo.performance_indicator.hv import Hypervolume

from pymoo.factory import get_decision_making, get_reference_directions


import detector


if __name__ == "__main__":

    infile = './survey_part1.pkl'
    #with open(infile, 'rb') as f:
    #    res = pickle.load(f)
    with open(infile, 'rb') as in_strm:
        l_res = dill.load(in_strm)

    res_X = l_res[0]
    res_F = l_res[1]



    plot = Scatter(labels=["tracker inefficiency","volume (proxy of cost)"])
    #plot = get_visualization("scatter")
    plot.add(res_F, color="red")
    #plot.saveas("test.pdf")
    plot.show()

    #--------------------------------------------------------------------------#

    dataset = pd.DataFrame({'f1': res_F[:, 0], 'f2': res_F[:, 1]})

    interesting_values = dataset.loc[(dataset["f1"]<0.25) & (dataset["f2"]<150)]
    print("interesting values: \n", interesting_values)


    #--------------------------------------------------------------------------#
    """
    # Select (Decision Making involved) a pair of values of f1, f2 from the Pareto front
    # Find corresponding values of design point (R, pitch, y1, y2, y3, z1, z2, z3)


    value_f1 = 0.153333
    value_f2 = 146.904501
    epsilon  = 0.0001

    idx_opt = dataset.loc[(dataset["f1"] > value_f1-epsilon*value_f1) & \
                      (dataset["f1"] < value_f1+epsilon*value_f1) & \
                      (dataset["f2"] > value_f2-epsilon*value_f2) & \
                      (dataset["f2"] < value_f2+epsilon*value_f2)].index

    print(idx_opt[0])
    idx_opt = idx_opt[0]

    X_par = res_X
    X_opt = X_par[idx_opt, :]

    print("... optimal point: \n", X_opt)
    print("... optimal value: \n", value_f1, value_f2)
    """
    #--------------------------------------------------------------------------#
    #                   PLOT CORRESPONDING GEOMETRY
    #--------------------------------------------------------------------------#
    """
    print(detector.y_min, detector.y_max)

    R, pitch, y1, y2, y3, z1, z2, z3 = X_opt
    tr = detector.Tracker(R, pitch, y1, y2, y3, z1, z2, z3)
    Z, Y = tr.create_geometry()
    num_wires = detector.calculate_wires(Y, detector.y_min, detector.y_max)

    volume = detector.wires_volume(Y, detector.y_min, detector.y_max,R)

    #print(".......display geometry....")
    #detector.geometry_display(Z, Y, R, y_min=detector.y_min, y_max=detector.y_max,block=False,pause=5) #5

    print("# of wires: ", num_wires, ", volume: ", volume)

    N_tracks = 150
    t = detector.Tracks(b_min=-100, b_max=100, alpha_mean=0, alpha_std=0.2)

    tracks = t.generate(N_tracks)

    print(".......display geometry with tracks....")
    detector.geometry_display(Z, Y, R, y_min=detector.y_min, y_max=detector.y_max,block=False, pause=-1)
    detector.tracks_display(tracks, Z,block=False,pause=9)
    """

    #--------------------------------------------------------------------------#
    #                           PSEUDO-WEIGHTS
    #--------------------------------------------------------------------------#

    # The obtained trade-off Pareto solutions can be assigned a pseudo-weight
    # (on each direction of the objexctive space, the pseudo-weight will range in [0,1])
    # where (1,0) means small f1 (x) and large f2 (y).
    # The values of the pseudo-weights will depend on the position of the points on the Pareto front
    # After a pseudo-weight is assigned to each solution, the one closer to the decision maker's wish may be selected

    weights_a = np.array([0., 0.])
    a, pseudo_weights_a = get_decision_making("pseudo-weights", weights_a).do(res_F, return_pseudo_weights=True)

    weights_b = np.array([0.5, 0.5])
    b, pseudo_weights_b = get_decision_making("pseudo-weights", weights_b).do(res_F, return_pseudo_weights=True)

    weights_c = np.array([1.0, 1.0])
    c, pseudo_weights_c = get_decision_making("pseudo-weights", weights_c).do(res_F, return_pseudo_weights=True)

    """
    plot = get_visualization("petal", bounds=(0, .5), reverse=True) # the boundaries for normalization purposes (does not apply for every plot
    # either 2d array [[min1,..minN],[max1,...,maxN]] or just two numbers [min,max]
    bounds=[0,1],
    plot.add(res_F[[a, b]])
    plot.show()
    """

    print(a)
    print(b)
    print(res_F[a])
    print(res_F[[a,b]])


    print("...weights: ", weights_a)
    #print("...pseudo_weights: ", pseudo_weights_a)
    X_par = res_X
    idx_opt = a
    X_opt = X_par[idx_opt, :]
    print("...solution and objectives")
    print(X_opt)
    print(res_F[idx_opt])

    print("")

    print("...weights: ", weights_b)
    #print("...pseudo_weights: ", pseudo_weights_b)
    idx_opt = b
    X_opt = X_par[idx_opt, :]
    print("...solution and objectives")
    print(X_opt)
    print(res_F[idx_opt])

    print("")

    print("...weights: ", weights_c)
    #print("...pseudo_weights: ", pseudo_weights_c)
    idx_opt = c
    X_opt = X_par[idx_opt, :]
    print("...solution and objectives")
    print(X_opt)
    print(res_F[idx_opt])
