import streamlit as st
import numpy as np
import random 
import string
import uuid
from scipy.stats import betaprime


st.set_page_config(page_title="Distribution Calculator")

st.title('Pathogen Distribution Calculator')
st.divider() 



def calculate_risk_exp(k, dose):
    risk = 1 - np.exp(-k * dose)
    return risk

def calculate_risk_beta_poisson_regular(alpha, beta, dose):
    risk = 1 - np.exp(-betaprime.pdf(dose, alpha, beta))
    return risk

def calculate_risk_beta_poisson_approximate(dose, alpha, param, n_50 = False):
    if n_50:
        risk = 1 - np.power((1 + dose * (np.power(2, 1/alpha)-1)/(param)), -alpha)
    else:
        param = param * (np.power(2, 1/ alpha) - 1)
        risk = 1 - np.power((1 + dose * (np.power(2, 1/alpha)-1)/(param)), -alpha)
    return risk

left_column, right_column = st.columns(2)


def display_exponential(identifier):
    st.subheader('Exponential Distribution')
    st.latex("1 - exp(-k \\times dose)")

    k_key = f'k_exp_{identifier}'
    dose_key = f'dose_exp_{identifier}'

    
    if not st.session_state:
        st.session_state[k_key] = 0.5
        st.session_state[dose_key] = 10.0

    
    elif k_key not in st.session_state or dose_key not in st.session_state:
        st.session_state[k_key] = 0.5
        st.session_state[dose_key] = 10.0

    k = st.number_input('Enter the value for k', key=k_key, min_value=0.0, max_value=1.0, value=st.session_state[k_key], step=0.000001)
    dose = st.number_input('Enter the value for dose', key=dose_key, min_value=0.0, max_value=100000.0, value=st.session_state[dose_key], step=0.000001)
    
    risk = calculate_risk_exp(k, dose)
    st.write(f'The calculated risk for exponential distribution is: {risk:.4f}')



def display_beta_poisson_regular(identifier):
    st.subheader('Beta-Poisson Distribution - Regular')
    st.latex("1 - [1 + \\frac{dose}{\\beta}]^{-\\alpha}")

    alpha_key = f'alpha_beta_{identifier}'
    beta_key = f'beta_beta_{identifier}'
    dose_key = f'dose_beta_{identifier}'

    
    if not st.session_state:
        st.session_state[alpha_key] = 1.0
        st.session_state[beta_key] = 1.0
        st.session_state[dose_key] = 10.0

    
    elif alpha_key not in st.session_state or beta_key not in st.session_state or dose_key not in st.session_state:
        st.session_state[alpha_key] = 1.0
        st.session_state[beta_key] = 1.0
        st.session_state[dose_key] = 10.0

    alpha = st.number_input('Enter the value for alpha', key=alpha_key, min_value=0.0, max_value=10.0, value=st.session_state[alpha_key], step=0.00000001)
    beta = st.number_input('Enter the value for beta', key=beta_key, min_value=0.0, max_value=10.0, value=st.session_state[beta_key], step=0.00000001)
    dose = st.number_input('Enter the dose value', key=dose_key, min_value=0.0, max_value=100.0, value=st.session_state[dose_key], step=0.00000001)
    
    risk = calculate_risk_beta_poisson_regular(alpha, beta, dose)
    st.write(f'The calculated risk for beta-poisson regular distribution is: {risk:.4f}')
def display_beta_poisson_approximate_beta(identifier):
    st.subheader('Beta-Poisson Distribution - Approximate (Beta)')
    st.latex("N_{50} = \\beta * [2^{\\frac{1}{\\alpha}} - 1]")
    st.latex("1 - [1 + dose \\times \\frac{(2^{\\frac{1}{\\alpha}} - 1)}{N_{50}}]^{-\\alpha}")

    alpha_key = f'alpha_beta_approx_{identifier}'
    beta_key = f'beta_beta_approx_{identifier}'
    dose_key = f'dose_beta_approx_{identifier}'

    if not st.session_state:
        st.session_state[alpha_key] = 1.0
        st.session_state[beta_key] = 1.0
        st.session_state[dose_key] = 10.0
    elif alpha_key not in st.session_state or beta_key not in st.session_state or dose_key not in st.session_state:
        st.session_state[alpha_key] = 1.0
        st.session_state[beta_key] = 1.0
        st.session_state[dose_key] = 10.0

    alpha = st.number_input('Enter the value for alpha', key=alpha_key, min_value=0.0, max_value=10.0, value=st.session_state[alpha_key], step=0.00000001)
    beta = st.number_input('Enter the value for beta', key=beta_key, min_value=0.0, max_value=10.0, value=st.session_state[beta_key], step=0.00000001)
    dose = st.number_input('Enter the dose value', key=dose_key, min_value=0.0, max_value=100.0, value=st.session_state[dose_key], step=0.00000001)

    risk = calculate_risk_beta_poisson_approximate(dose, alpha, beta, False)
    st.write(f'The calculated risk for beta-poisson approximate (Beta) distribution is: {risk:.4f}')

def display_beta_poisson_approximate_n50(identifier):
    st.subheader('Beta-Poisson Distribution - Approximate (N_50)')
    st.latex("1 - [1 + dose \\times \\frac{(2^{\\frac{1}{\\alpha}} - 1)}{N_{50}}]^{-\\alpha}")

    alpha_key = f'alpha_n50_approx_{identifier}'
    n50_key = f'n50_n50_approx_{identifier}'
    dose_key = f'dose_n50_approx_{identifier}'

    if not st.session_state:
        st.session_state[alpha_key] = 1.0
        st.session_state[n50_key] = 1.0
        st.session_state[dose_key] = 10.0
    elif alpha_key not in st.session_state or n50_key not in st.session_state or dose_key not in st.session_state:
        st.session_state[alpha_key] = 1.0
        st.session_state[n50_key] = 1.0
        st.session_state[dose_key] = 10.0

    alpha = st.number_input('Enter the value for alpha', key=alpha_key, min_value=0.0, max_value=10.0, value=st.session_state[alpha_key], step=0.00000001)
    n_50 = st.number_input('Enter the value for N50', key=n50_key, min_value=0.0, max_value=10.0, value=st.session_state[n50_key], step=0.00000001)
    dose = st.number_input('Enter the dose value', key=dose_key, min_value=0.0, max_value=100.0, value=st.session_state[dose_key], step=0.00000001)

    risk = calculate_risk_beta_poisson_approximate(dose, alpha, n_50, True)
    st.write(f'The calculated risk for beta-poisson approximate (N_50) distribution is: {risk:.4f}')

def display_selection(key):
    selection = st.selectbox("Choose a Distribution",
        ("Exponential Distribution", "Beta-Poisson Distribution - Regular", "Beta-Poisson Distribution - Approximate (Beta)",
         "Beta-Poisson Distribution - Approximate (N_50)"),
        key=key)
    if selection == "Exponential Distribution":
        display_exponential(key)
    elif selection == "Beta-Poisson Distribution - Regular":
        display_beta_poisson_regular(key)
    elif selection == "Beta-Poisson Distribution - Approximate (Beta)":
        display_beta_poisson_approximate_beta(key)
    elif selection == "Beta-Poisson Distribution - Approximate (N_50)":
        display_beta_poisson_approximate_n50(key)

with left_column:
    st.header("Box 1")
    display_selection(key="select_1")    
    st.divider() 
        


with right_column:
    st.header("Box 2")
    display_selection(key="select_2")
    st.divider() 
        

left_column_2, right_column_2 = st.columns(2)
with left_column_2:
    st.header("Box 3")
    display_selection(key="select_3")


with right_column_2:
    st.header("Box 4")
    display_selection(key="select_4")
