#---- CF (ongoing) ML WS
import os
os.environ['MPLCONFIGDIR'] = "/tmp/"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import detector2
import re
import pickle
import dill

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.model.problem import Problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.factory import get_visualization, get_decomposition
from pymoo.util.display import Display

from pymoo.factory import get_performance_indicator
from pymoo.performance_indicator.hv import Hypervolume

#from utils import *


rand_st = np.random.randint(1,10000)
rand_st = 1317 #for reproducibility

# CONSTANT PARAMETERS
R = .5  # cm
pitch = 10.0  #cm
ncalls = 10
y_min=-10.1
y_max=10.1

# ADJUSTABLE PARAMETERS
y1 = 0.0
y2 = 0.0
y3 = 0.0
z1 = 2.0
z2 = 4.0
z3 = 6.0

#------------- GEOMETRY ---------------#
print(".....INITIAL GEOMETRY")
tr = detector2.Tracker(R, pitch, y1, y2, y3, z1, z2, z3)
Z, Y = tr.create_geometry()
num_wires = detector2.calculate_wires(Y, y_min, y_max)

volume = detector2.wires_volume(Y, y_min, y_max,R)

detector2.geometry_display(Z, Y, R, y_min=y_min, y_max=y_max,block=False,pause=5) #5

print("# of wires: ", num_wires, ", volume: ", volume)

N_tracks = 1000
t = detector2.Tracks(b_min=-100, b_max=100, alpha_mean=0, alpha_std=0.2)
tracks = t.generate(N_tracks)

detector2.geometry_display(Z, Y, R, y_min=y_min, y_max=y_max,block=False, pause=-1)
detector2.tracks_display(tracks, Z,block=False,pause=5)

 #a track is detected if at least two wires have been hit
score = detector2.get_score(Z, Y, tracks, R)
print("fraction of tracks detected: ",score)





class objectives():

  def __init__(self,tracks,y_min,y_max):
    self.tracks = tracks
    self.y_min = y_min
    self.y_max = y_max

  def wrapper_geometry(fun):

      def inner(self):
          R, pitch, y1, y2, y3, z1, z2, z3 = self.X
          self.geometry(R, pitch, y1, y2, y3, z1, z2, z3)
          return fun(self)
      return inner

  def update_tracks(self, new_tracks):
    self.tracks = new_tracks

  def update_design_point(self,X):
      self.X = X


  def geometry(self,R, pitch, y1, y2, y3, z1, z2, z3):
    tr = detector2.Tracker(R, pitch, y1, y2, y3, z1, z2, z3)
    self.R = R
    self.Z, self.Y = tr.create_geometry()


  @wrapper_geometry
  def calc_score(self):
      res = detector2.get_score(self.Z, self.Y, self.tracks, self.R)
      return res


  def get_score(self,X):
    R, pitch, y1, y2, y3, z1, z2, z3 = X
    self.geometry(R, pitch, y1, y2, y3, z1, z2, z3)
    res = detector2.get_score(self.Z, self.Y, self.tracks, self.R)
    return res


  def get_volume(self):
    volume = detector2.wires_volume(self.Y, self.y_min, self.y_max,self.R)
    return volume



res = objectives(tracks,y_min,y_max)

#res.geometry(R, pitch, y1, y2, y3, z1, z2, z3)

X = R, pitch, y1, y2, y3, z1, z2, z3
#fscore  = res.get_score(X)
res.update_design_point(X)
fscore  = res.calc_score()
fvolume = res.get_volume()

print("...check: ", fvolume, fscore)

#--------------------------------------------------------------------#
#-----------------------------------#
#        Class Definitions          #
#-----------------------------------#
#---------------------------------------------------------------------

"""
R_min, R_max          = [0.5, 1.0]
pitch_min, pitch_max  = [2.5, 5.0]
y1_min, y1_max        = [0., 4.]
y2_min, y2_max        = [0., 4.]
y3_min, y3_max        = [0., 4.]
z1_min, z1_max        = [2., 10.]
z2_min, z2_max        = [2., 10.]
z3_min, z3_max        = [2., 10.]
"""

class MyProblem(Problem):

    #--------- vectorized ---------#

    def __init__(self):
        super().__init__(n_var=8,
                         n_obj=2,
                         n_constr=0,
                         xl=np.array([0.5,2.5,0.,0.,0.,2.,2.,2.]),
                         xu=np.array([1.0,5.0,4.,4.,4.,10.,10.,10.]),elementwise_evaluation=True)

    def _evaluate(self, x, out, *args, **kwargs):

        #detector.py
        #f1 = 1.-res.get_score(x)
        #f2 = res.get_volume()
        #detector2.py
        f1 = 1.-res.get_score(x)[0]
        f2 = res.get_volume()
        #f1 = fu1(x)
        #f2 = fu2(x)
        #g1 = gu1(x)
        #g2 = gu2(x)

        #f1 = x[0] ** 2 + x[1] ** 2
        #f2 = (x[0] - 1) ** 2 + x[1] ** 2

        #g1 = 2 * (x[0] - 0.1) * (x[0] - 0.9) / 0.18
        #g2 = - 20 * (x[0] - 0.4) * (x[0] - 0.6) / 4.8

        out["F"] = [f1, f2]
        #out["G"] = [g1, g2]




#--------------------------------------------------------------------#
#-----------------------------------#
#          Optimization             #
#-----------------------------------#


problem = MyProblem()



#https://pymoo.org/getting_started.html
algorithm = NSGA2(pop_size=100,n_offsprings=20) #n_offsprings=10

res = minimize(problem,
               algorithm,
               ("n_gen", 100),
               verbose=True,
               seed=1,
               save_history=True)

               #display=MyDisplay()) you can choose what to print

               #save_history with a deepcopy can be memory intensive
               #used for plotting convergence.
               #For MOO, report HyperVolume + Constraint Violation




#fig = plt.figure()
#timer = fig.canvas.new_timer(interval = 20000) #creating a timer object and setting an interval of 20000 milliseconds
#timer.add_callback(detector.close_event)
#timer.start()


#plot = Scatter()
plot = get_visualization("scatter")
plot.add(res.F, color="red")
#plot.saveas("test.pdf")
plot.show()
#plot.pause(10)


#-----------------------------------#
#          Post-Processing          #
#-----------------------------------#
#------------------------------------------------------------------------------#

n_evals = []    # corresponding number of function evaluations\
F = []          # the objective space values in each generation
cv = []         # constraint violation in each generation


# iterate over the deepcopies of algorithms
for algorithm in res.history:

    # store the number of function evaluations
    n_evals.append(algorithm.evaluator.n_eval)

    # retrieve the optimum from the algorithm
    opt = algorithm.opt

    # store the least contraint violation in this generation
    cv.append(opt.get("CV").min())

    # filter out only the feasible and append
    feas = np.where(opt.get("feasible"))[0]
    _F = opt.get("F")[feas]
    F.append(_F)



#-----------------------------------#
#              Metrics              #
#-----------------------------------#
#------------------------------------------------------------------------------#

# MODIFY - this is problem dependend
ref_point = np.array([0.5, 200])

# create the performance indicator object with reference point
metric = Hypervolume(ref_point=ref_point, normalize=False)


# calculate for each generation the HV metric
hv = [metric.calc(f) for f in F]  #or res.F

# visualze the convergence curve
plt.plot(n_evals, hv, '-o', markersize=4, linewidth=2)
plt.title("Convergence")
plt.xlabel("Function Evaluations")
plt.ylabel("Hypervolume")
plt.show()


# visualze the convergence curve
plt.plot(n_evals, cv, '-o', markersize=4, linewidth=2)
plt.title("Convergence")
plt.xlabel("Function Evaluations")
plt.ylabel("Constraint Violation")
plt.show()


#-----------------------------------#
#          Decision Making          #
#-----------------------------------#
#------------------------------------------------------------------------------#


weights = np.array([1.0, 0.]) #0.5, 0.5   #sembra ininfluente?
decomp = get_decomposition("asf")


#print(np.shape(F))
#print(F)

F_new = np.concatenate(F, axis=0 )

#F_new = F_new.reshape(len(F),2)

#print(F_new)

#print(np.shape(F_new))



I = get_decomposition("asf").do(F_new, weights).argmin()
print("Best regarding decomposition: Point %s - %s" % (I, F_new[I]))

arx = F_new[I][0]
ary = F_new[I][1]

#print(arrow_l)
#print(type(arrow_l))
#print(np.shape(arrow_l))

#exit()

#arrow_l = arrow_l.split()

#exit()
#arx = re.sub("[^\d\.]", "", arrow_l[1])
#ary = re.sub("[^\d\.]", "", arrow_l[2])

plot = get_visualization("scatter")
plot.add(F_new, color="blue", alpha=0.2, s=10)
plot.add(F_new[I], color="red", s=30)
plot.do()
plot.apply(lambda ax: ax.arrow(0, 0, arx, ary, color='black',
                               head_width=0.01, head_length=0.01, alpha=0.4))
plot.show()

# https://pymoo.org/interface/result.html

print(res.X)
print(np.shape(res.X))


print(res.F)
print(np.shape(res.F))

print(res.pop.get("X"))
print(np.shape(res.pop.get("X")))


#--------------------------#
#      store res           #
#--------------------------#
#------------------------------------------------------------------------------#


out_t_filename = "./survey_part1.pkl" #'../strategy_comparison/gp_time.pkl'


with open(out_t_filename,'wb') as multi_file_survey:
    out_list = [res.X, res.F]
    dill.dump(out_list, multi_file_survey)

exit()

"""

dataset = pd.DataFrame({'f1': res.F[:, 0], 'f2': res.F[:, 1]})

interisting_values = dataset.loc[(dataset["f1"]<0.25)]
print(interisting_values)

"""

"""
value_f1 = 0.64
value_f2 = 53.38
epsilon  = 0.00000001

idx_opt = dataset.loc[(dataset["f1"] > value_f1-epsilon*value_f1) & \
                  (dataset["f1"] < value_f1+epsilon*value_f1) & \
                  (dataset["f2"] > value_f2-epsilon*value_f2) & \
                  (dataset["f2"] < value_f2+epsilon*value_f2)].index

print(idx_opt[0])
idx_opt = idx_opt[0]

X_par = res.X
X_opt = X_par[idx_opt, :]

print("... optimal point: \n", X_opt)
print("... optimal value: \n", value_f1, value_f2)
"""
