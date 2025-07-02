import numpy as np
from scipy.signal import savgol_filter
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

# TODO: Compute partial derivative of the input Ft, Fn, Fa 
# Desired output: [dFx, dFy, dFz]
def compute_smoothed_diff(Ft, Fn, Fa, time_step, window_length=5,polyorder=3):

    '''
    compute smoothed derivatives of Ft,Fn,Fa using Savitzky-Golay filter.
    Step1: smooth data by Savitzky-golay filter
    Step2: initialize derivative
    Step3: compute central difference
    Step4: compute derivative at boundary points
    '''
    df=[]
    for f in [Ft,Fn,Fa]:
        smoothed_data=savgol_filter(f,window_length=window_length, polyorder=polyorder)
        f_diff=np.zeros_like(f)

        for i in range(1,len(f_diff)-2):
            f_diff[i]=(smoothed_data[i+1]-smoothed_data[i-1])/(2.0*(time_step[i-1]-time_step[i+1]))
            
        f_diff[0]=(smoothed_data[1]-smoothed_data[0])/(time_step[0]-time_step[1])
        f_diff[-1]=(smoothed_data[-1]-smoothed_data[-2])/(time_step[-2]-time_step[-1])

        df.append(f_diff)
    return df