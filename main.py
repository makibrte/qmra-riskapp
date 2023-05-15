import streamlit as st
import numpy as np
from scipy.stats import betaprime

st.title('Distribution Calculator')

# Define functions to calculate risk for each distribution
def calculate_risk_exp(k, dose):
    risk = 1 - np.exp(-k * dose)
    return risk

def calculate_risk_beta_poisson(alpha, beta, dose):
    risk = 1 - np.exp(-betaprime.pdf(dose, alpha, beta))
    return risk

# Use a selectbox to choose the distribution
distributions = st.select_slider(
    'Select a color of the rainbow',
    options=['Exponential', 'Beta-Poisson'])
#distribution = st.selectbox('Choose a distribution', ('Exponential', 'Beta-Poisson'))

if distributions == 'Exponential':
    # Use sliders to select parameters for exponential distribution
    k = st.slider('Enter the value for k', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    dose = st.slider('Enter the value for dose', min_value=0.0, max_value=100.0, value=10.0, step=0.1)

    if st.button('Calculate Risk for Exponential'):
        # Calculate risk
        risk = calculate_risk_exp(k, dose)

        # Display risk
        st.write(f'The calculated risk for exponential distribution is: {risk:.4f}')

elif distributions == 'Beta-Poisson':
    # Use sliders to select parameters for beta-poisson distribution
    alpha = st.slider('Enter the value for alpha', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    beta = st.slider('Enter the value for beta', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    dose = st.slider('Enter the value for dose', min_value=0.0, max_value=100.0, value=10.0, step=0.1)

    if st.button('Calculate Risk for Beta-Poisson'):
        # Calculate risk
        risk = calculate_risk_beta_poisson(alpha, beta, dose)

        # Display risk
        st.write(f'The calculated risk for beta-poisson distribution is: {risk:.4f}')
