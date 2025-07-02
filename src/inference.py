import torch
import numpy as np
import sympy
from models.nonlinear_regression import NonlinearRegressionModel

def inference(input, model_path):
    model = NonlinearRegressionModel()
    model.load_state_dict(torch.load(model_path))

    # Post process function 
    K, vc, ft, ap, ad = sympy.symbols('K vc ft ap ad')
    weights = model.state_dict()["weights"].numpy()

    for i, w in enumerate(weights):
        w = np.exp(w)
        expression = K * vc**w[2] * ft**w[3] * ap**w[4] * ad**w[5]
        latex_str = sympy.latex(expression, mode='inline')

        if i == 0:
            print(f"Ft = {latex_str}")
        elif i == 1:
            print(f"Fn = {latex_str}")
        elif i == 2:
            print(f"Fa = {latex_str}")

if __name__ == "__main__":
    inference(None, "../models/test.pth")
