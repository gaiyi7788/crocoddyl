import os
import signal
import sys
import time

import example_robot_data
import numpy as np
import pinocchio

import crocoddyl
from crocoddyl.utils.biped import SimpleBipedGaitProblem, plotSolution

WITHDISPLAY = "display" in sys.argv or "CROCODDYL_DISPLAY" in os.environ
WITHPLOT = "plot" in sys.argv or "CROCODDYL_PLOT" in os.environ
signal.signal(signal.SIGINT, signal.SIG_DFL)

# Creating the lower-body part of Talos
talos_legs = example_robot_data.load("zj_humanoid")
# 这里我修改了zj_biped的urdf中的effort的上限，原先只允许effort="2"，现在修改成了effort="200"
lims = talos_legs.model.effortLimit
lims *= 0.5  # reduced artificially the torque limits
talos_legs.model.effortLimit = lims

# Defining the initial state of the robot
# q0 = talos_legs.model.referenceConfigurations["half_sitting"].copy()
q0 = np.array(
    [
        0,0,0.6,
        0,0,0,1,
        0,0,0,0,
        0,0,0,0,
        0,0,0,0,0,0,
        0,0,0,0,0,0,
    ]
)

# 1.27061782e-01,  2.07969163e-02,  5.82793935e-01,  
# 4.82742869e-02,  7.23105979e-02, -4.13900879e-01, 9.06160490e-01, 
# -1.79680331e-07, 3.86983906e-02,  6.81113258e-05, -6.89665975e-06, 
# -2.00078150e-01, 6.82578132e-05, -6.82705198e-06, -2.00078122e-01, 
# -1.44525990e-05, -6.27852612e-05, -5.40004061e-01, 8.18200235e-01, -4.50044821e-01, -8.21892894e-05, 
# -4.25584796e-05, -1.84876418e-04, -5.40019801e-01, 8.18366666e-01, -4.50137404e-01, -2.70846453e-04


talos_legs.model.referenceConfigurations["half_sitting"] = q0
v0 = pinocchio.utils.zero(talos_legs.model.nv)
x0 = np.concatenate([q0, v0])

# Setting up the 3d walking problem
# 这里在urdf中增加的fixed_link, 其父关节是leg的link 6
rightFoot = "right_sole_link"
leftFoot = "left_sole_link"

gait = SimpleBipedGaitProblem(talos_legs.model, rightFoot, leftFoot)

# Setting up all tasks
GAITPHASES = [
    {
        "walking": {
            "stepLength": 0.6,
            "stepHeight": 0.1,
            "timeStep": 0.03,
            "stepKnots": 35,
            "supportKnots": 10,
        }
    },
    {
        "walking": {
            "stepLength": 0.6,
            "stepHeight": 0.1,
            "timeStep": 0.03,
            "stepKnots": 35,
            "supportKnots": 10,
        }
    },
    {
        "walking": {
            "stepLength": 0.6,
            "stepHeight": 0.1,
            "timeStep": 0.03,
            "stepKnots": 35,
            "supportKnots": 10,
        }
    },
    {
        "walking": {
            "stepLength": 0.6,
            "stepHeight": 0.1,
            "timeStep": 0.03,
            "stepKnots": 35,
            "supportKnots": 10,
        }
    },
]
cameraTF = [3.0, 3.68, 0.84, 0.2, 0.62, 0.72, 0.22]

solver = [None] * len(GAITPHASES)
display = None
if WITHDISPLAY:
    if display is None:
        try:
            import gepetto

            gepetto.corbaserver.Client()
            display = crocoddyl.GepettoDisplay(
                talos_legs, 4, 4, cameraTF, frameNames=[rightFoot, leftFoot]
            )
        except Exception:
            display = crocoddyl.MeshcatDisplay(
                talos_legs, frameNames=[rightFoot, leftFoot]
            )
for i, phase in enumerate(GAITPHASES):
    for key, value in phase.items():
        if key == "walking":
            # Creating a walking problem
            solver[i] = crocoddyl.SolverBoxFDDP(
                gait.createWalkingProblem(
                    x0,
                    value["stepLength"],
                    value["stepHeight"],
                    value["timeStep"],
                    value["stepKnots"],
                    value["supportKnots"],
                )
            )
            solver[i].th_stop = 1e-7

    # Added the callback functions
    print("*** SOLVE " + key + " ***")
    if WITHDISPLAY and type(display) == crocoddyl.GepettoDisplay:
        if WITHPLOT:
            solver[i].setCallbacks(
                [
                    crocoddyl.CallbackVerbose(),
                    crocoddyl.CallbackLogger(),
                    crocoddyl.CallbackDisplay(display),
                ]
            )
        else:
            solver[i].setCallbacks(
                [crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)]
            )
    elif WITHPLOT:
        solver[i].setCallbacks(
            [
                crocoddyl.CallbackVerbose(),
                crocoddyl.CallbackLogger(),
            ]
        )
    else:
        solver[i].setCallbacks([crocoddyl.CallbackVerbose()])
    solver[i].getCallbacks()[0].precision = 3
    solver[i].getCallbacks()[0].level = crocoddyl.VerboseLevel._2

    # Solving the problem with the DDP solver
    xs = [x0] * (solver[i].problem.T + 1)
    us = solver[i].problem.quasiStatic([x0] * solver[i].problem.T)
    solver[i].solve(xs, us, 100, False, 0.1)

    # Defining the final state as initial one for the next phase
    x0 = solver[i].xs[-1]

# Display the entire motion
if WITHDISPLAY:
    display.rate = -1
    display.freq = 1
    while True:
        for i, phase in enumerate(GAITPHASES):
            display.displayFromSolver(solver[i])
        time.sleep(1.0)

# Plotting the entire motion
if WITHPLOT:
    plotSolution(solver, bounds=False, figIndex=1, show=False)

    for i, phase in enumerate(GAITPHASES):
        title = list(phase.keys())[0] + " (phase " + str(i) + ")"
        log = solver[i].getCallbacks()[1]
        crocoddyl.plotConvergence(
            log.costs,
            log.u_regs,
            log.x_regs,
            log.grads,
            log.stops,
            log.steps,
            figTitle=title,
            figIndex=i + 3,
            show=True if i == len(GAITPHASES) - 1 else False,
        )
