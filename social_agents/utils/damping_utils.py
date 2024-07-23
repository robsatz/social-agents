import numpy as np

def calculate_damping(a, proximity, thres=5):
    """calculate the damping coefficient

    Parameters
    ----------
    a : float
        the effect of proximity to visicosity
    proximity : float
            the distance between agents
    thres : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    if proximity <= thres:
        return a*proximity
    else:
        return 1

def set_joint_damping(physics, damping_value):
    """setting damping to the DeepMind Swimmer

    Parameters
    ----------
    physics : mujuco.physics
        an object describing the swimmer
    damping_value : float
        damping coefficient
    """
    for joint_id in range(len(physics.model.dof_damping)):
        physics.model.dof_damping[joint_id] = damping_value



def NCAP_damping(delt, a=0.0001, proximity=1, thres=5):
        """return a damping value 

        Parameters
        ----------
        a : float
            the effect of proximity to visicosity
        proximity : float
            the distance between agents
        thres: float
            the threshold distance where there is no effect
        delt: time to where damping occurs
        """

        if proximity <= thres:
            _lamda = calculate_damping(a, proximity, thres)
            return np.exp(-_lamda*delt)