import numpy as np


def calculate_risk_exp(k=0, dose=0, risk=0, for_dose = False):
    
        
    
    if for_dose:
        dose = -np.log(1-risk) / k
        return dose
    else:
        risk = 1 - np.exp(-k * dose)
        return risk

def calculate_risk_beta_poisson_regular(alpha=0.0, beta=0.0, dose=0.0, risk=0.0, for_dose = False):
    if for_dose == False:
        risk = 1 - np.power(1 + dose/beta, -alpha)
        return risk
    else:
        dose = beta*np.power(1-risk, -1/alpha) - 1
        return dose

def calculate_risk_beta_poisson_approximate(dose, alpha, param, n_50 = False, risk = 0, for_dose = False):
    if for_dose == False:
        if n_50:
            risk = 1 - np.power((1 + dose * (np.power(2, 1/alpha)-1)/(param)), -alpha)
        else:
            param = param * (np.power(2, 1/ alpha) - 1)
            risk = 1 - np.power((1 + dose * (np.power(2, 1/alpha)-1)/(param)), -alpha)
        return risk
    else:
        if n_50:
            dose = - (np.power((risk-1), -alpha) + 1)/((np.power(2, 1/alpha) - 1)/ (param))
            return dose