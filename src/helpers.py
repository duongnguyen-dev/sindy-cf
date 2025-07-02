import numpy as np

def compute_cutting_foce(channels: list):
    Fx = channels[0] + channels[1]
    Fy = channels[2] + channels[3]
    Fz = sum(channels[4:])
    
    return Fx, Fy, Fz

def convert_force_to_rotating_tool_frame(Fx, Fy, Fz, time_step, rpm=2000):
    '''
    This function will help to convert the force to the rotating tool frame where:
        Ft: tangential force
        Fr: radial force
        Fa: axial force
        time_step: time step of the data
        rpm: rotation speed per minute of the tool
    '''

    theta = 2 * np.pi * (rpm / 60) * time_step

    Ft = Fx * np.cos(theta) + Fy * np.sin(theta)
    Fn = Fx * np.sin(theta) - Fy * np.cos(theta)
    Fa = Fz

    return Ft, Fn, Fa

# TODO: Compute partial derivative of the input Fx, Fy, Fz 
# Desired output: [dFx, dFy, dFz]
def compute_smoothed_diff(Ft, Fn, Fa): 
    '''
    cutting_force: this parameter should be a list with 3 elements Fx, Fy, Fz
    '''
    return [1, 1, 1] # Just dum return for testing