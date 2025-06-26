def compute_cutting_foce(channels: list):
    Fx = channels[0] + channels[1]
    Fy = channels[2] + channels[3]
    Fz = sum(channels[4:])
    
    return Fx, Fy, Fz

# TODO: Compute partial derivative of the input Fx, Fy, Fz 
# Desired output: [dFx, dFy, dFz]
def compute_smoothed_diff(cutting_force: list): 
    '''
    cutting_force: this parameter should be a list with 3 elements Fx, Fy, Fz
    '''
    pass