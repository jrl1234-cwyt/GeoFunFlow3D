import math


def get_mu_fae(epoch: int, total_epochs: int, target_mu: float = 5e-6):

    tau = epoch / total_epochs
    tau_1 = 0.20  
    tau_2 = 0.50  

    if tau < tau_1:
 
        return 0.0
    elif tau > tau_2:

        return target_mu
    else:
        scale = (tau - tau_1) / (tau_2 - tau_1)
        smooth_scale = 0.5 * (1.0 - math.cos(math.pi * scale))
        return target_mu * smooth_scale


def get_lambda_flow(epoch: int, max_epoch: int, base: float = 5e-5, end_scale: float = 0.1):
  
    if max_epoch <= 0:
        return base

    tau = max(0.0, min(1.0, epoch / max_epoch))

    return base * (end_scale ** tau)
