from crocoddyl import StatePinocchio
from crocoddyl import DifferentialActionModelFloatingInContact
from crocoddyl import IntegratedActionModelEuler
from crocoddyl import CostModelSum
from crocoddyl import CostModelFramePlacement, CostModelFrameVelocity
from crocoddyl import CostModelState, CostModelControl
from crocoddyl import ActivationModelWeightedQuad
from crocoddyl import ActuationModelFreeFloating
from crocoddyl import ContactModel6D, ContactModelMultiple
from crocoddyl import ShootingProblem
from crocoddyl import SolverDDP
from crocoddyl import CallbackDDPLogger, CallbackDDPVerbose, CallbackSolverDisplay
from crocoddyl import plotOCSolution, plotDDPConvergence
from crocoddyl import loadTalosLegs
from crocoddyl import m2a, a2m
import pinocchio
from pinocchio.utils import *

robot = loadTalosLegs()
robot.model.armature[6:] = 1.
robot.initDisplay(loadModel=True)

rmodel = robot.model
rdata  = rmodel.createData()

# Setting up the 3d walking problem
rightFoot = 'right_sole_link'
leftFoot = 'left_sole_link'

rightId = rmodel.getFrameId(rightFoot)
leftId  = rmodel.getFrameId(leftFoot)

# Create the initial state
q0 = robot.q0.copy()
v0 = zero(rmodel.nv)
x0 = m2a(np.concatenate([q0,v0]))
rmodel.defaultState = x0.copy()

# Solving the 3d walking problem using DDP
stepLength = 0.2
swingDuration = 0.75
stanceDurantion = 0.1


def runningModel(contactIds, effectors, integrationStep = 1e-2):
    '''
    Creating the action model for floating-base systems. A walker system 
    is by default a floating-base system.
    contactIds is a list of frame Ids of points that should be in contact.
    effectors is a dict of key frame ids and SE3 values of effector references.
    '''
    actModel = ActuationModelFreeFloating(rmodel)
    State = StatePinocchio(rmodel)

    # Creating a 6D multi-contact model, and then including the supporting foot
    contactModel = ContactModelMultiple(rmodel)
    for cid in contactIds:
       contactModel.addContact('contact%d'%cid,
                               ContactModel6D(rmodel, cid, ref=None))

    # Creating the cost model for a contact phase
    costModel = CostModelSum(rmodel, actModel.nu)
    wx = np.array([0]*6 + [.1]*(rmodel.nv-6) + [10]*rmodel.nv)
    #xneutral = np.concatenate([m2a(rmodel.neutralConfiguration),np.zeros(rmodel.nv)])
    costModel.addCost('xreg',weight=1e-1,
                      cost=CostModelState(rmodel,State,ref=rmodel.defaultState,nu=actModel.nu,
                                          activation=ActivationModelWeightedQuad(wx)))
    costModel.addCost('ureg',weight=1e-4,
                      cost=CostModelControl(rmodel, nu=actModel.nu))
    for fid,ref in effectors.items():
        costModel.addCost("track%d"%fid, weight=100.,
                          cost = CostModelFramePlacement(rmodel,fid,ref,actModel.nu))
    
    # Creating the action model for the KKT dynamics with simpletic Euler
    # integration scheme
    dmodel = \
             DifferentialActionModelFloatingInContact(rmodel,
                                                      actModel,
                                                      contactModel,
                                                      costModel)
    model = IntegratedActionModelEuler(dmodel)
    model.timeStep = integrationStep
    return model

def pseudoImpactModel(contactIds,effectors):
    assert(len(effectors)==1)
    model = runningModel(contactIds,effectors,integrationStep=0)

    costModel = model.differential.costs
    for fid,ref in effectors.items():
        costModel.addCost('impactVel',weight = 10000.,
                          cost=CostModelFrameVelocity(rmodel,fid))
        costModel.costs['track%d'%fid ].weight = 100000
    costModel.costs['xreg'].weight = 1
    costModel.costs['ureg'].weight = 0.01

    return model

SE3 = pinocchio.SE3
pinocchio.forwardKinematics(rmodel,rdata,q0)
pinocchio.updateFramePlacements(rmodel,rdata)
right0 = rdata.oMf[rightId].translation


models =\
         [    runningModel([ rightId, leftId ],{}, integrationStep=5e-2),]*10 \
         +  [ runningModel([ leftId ],{ rightId: SE3(eye(3), right0+np.matrix([0,0,0.1]).T)},
                           integrationStep=5e-2) ]*20 \
         +  [ runningModel([ leftId ],{ rightId: SE3(eye(3), right0+np.matrix([0,0,.0]).T)  },
                           integrationStep=5e-2) ]*20 \
         +  [ pseudoImpactModel([ leftId ],{ rightId: SE3(eye(3), right0+np.matrix([0,0,.0]).T)  }) ] \
         +  [ runningModel([ rightId, leftId ],{}) ]
                  
problem = ShootingProblem(initialState=x0,runningModels=models[:50],terminalModel=models[50])
ddp = SolverDDP(problem)
ddp.callback = [ CallbackDDPLogger(), CallbackDDPVerbose() ]
ddp.th_stop = 1e-9
ddp.solve(maxiter=1000,regInit=.1,init_xs=[rmodel.defaultState]*len(ddp.models()))


'''
    # Computing the time step per each contact phase given the step duration.
    # Here we assume a constant number of knots per phase
    numKnots = 20
    timeStep = float(stepDuration)/numKnots

        # Getting the frame id for the right and left foot
        rightFootId = self.robot.model.getFrameId(self.rightFoot)
        leftFootId = self.robot.model.getFrameId(self.leftFoot)

        # Compute the current foot positions
        q0 = a2m(x[:self.robot.nq])
        rightFootPos0 = self.robot.framePlacement(q0, rightFootId).translation
        leftFootPos0 = self.robot.framePlacement(q0, leftFootId).translation

        # Defining the action models along the time instances
        n_cycles = 2
        loco3dModel = []
        import copy
        for i in range(n_cycles):
            # swing LF phase
            leftSwingModel = \
                [ self.createContactPhaseModel(
                    timeStep,
                    rightFootId,
                    TaskSE3(
                        pinocchio.SE3(np.eye(3),
                                      np.asmatrix(a2m([ [(stepLength*k)/numKnots, 0., 0.] ]) +
                                      leftFootPos0)),
                        leftFootId)
                    ) for k in range(numKnots) ]
            
            # Double support phase
            doubleSupportModel = \
                self.createContactSwitchModel(
                    rightFootId,
                    TaskSE3(
                        pinocchio.SE3(np.eye(3),
                                      np.asmatrix(a2m([ stepLength, 0., 0. ]) +
                                      leftFootPos0)),
                        leftFootId)
                    )
            
            # swing RF phase
            rightSwingModel = \
                [ self.createContactPhaseModel(
                    timeStep,
                    leftFootId,
                    TaskSE3(
                        pinocchio.SE3(np.eye(3),
                                      np.asmatrix(a2m([ 2*(stepLength*k)/numKnots, 0., 0. ]) +
                                      rightFootPos0)),
                        rightFootId)
                    ) for k in range(numKnots) ]
            
            # Final support phase
            finalSupport = \
                self.createContactSwitchModel(
                    leftFootId,
                    TaskSE3(
                        pinocchio.SE3(np.eye(3),
                                      np.asmatrix(a2m([ 2*stepLength, 0., 0. ]) +
                                      rightFootPos0)),
                        rightFootId),
                    )
            rightFootPos0 += np.asmatrix(a2m([ stepLength, 0., 0. ]))
            leftFootPos0 += np.asmatrix(a2m([ stepLength, 0., 0. ]))
            loco3dModel += leftSwingModel + [ doubleSupportModel ] + rightSwingModel + [ finalSupport ]

        problem = ShootingProblem(x, loco3dModel, finalSupport)
        return problem

    def createContactSwitchModel(self, contactFootId, swingFootTask):
        model = self.createContactPhaseModel(0., contactFootId, swingFootTask)

        impactFootVelCost = \
            CostModelFrameVelocity(self.robot.model, swingFootTask.frameId)
        model.differential.costs.addCost('impactVel', impactFootVelCost, 10000.)
        model.differential.costs['impactVel' ].weight = 100000
        model.differential.costs['footTrack' ].weight = 100000
        model.differential.costs['stateReg'].weight = 1
        model.differential.costs['ctrlReg'].weight = 0.01
        return model









'''
