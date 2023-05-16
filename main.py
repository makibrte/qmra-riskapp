import streamlit as st
import numpy as np
from scipy.stats import betaprime

st.title('Distribution Calculator')

# Define functions to calculate risk for each distribution
def calculate_risk_exp(k, dose):
    risk = 1 - np.exp(-k * dose)
    return risk

def calculate_risk_beta_poisson_regular(alpha, beta, dose):
    risk = 1 - np.exp(-betaprime.pdf(dose, alpha, beta))
    return risk
def calculate_risk_beta_poisson_approximate(dose, alpha, param ,n_50 = False):
    if n_50:
        risk = 1 - np.power((1 + dose * (np.power(2, 1/alpha)-1)/(param)), -alpha)
        return risk
    else:
        risk = 1 - np.power((1 + dose / param), -alpha)
        return risk

# Use a selectbox to choose the distribution
distributions = st.select_slider(
    'Choose a distribution',
    options=['Exponential', 'Beta-Poisson'])
#distribution = st.selectbox('Choose a distribution', ('Exponential', 'Beta-Poisson'))

if distributions == 'Exponential':
    st.latex("1 - (-k \\times dose)")
    # Use sliders to select parameters for exponential distribution
    k = st.number_input('Enter the value for k', min_value=0.0, max_value=1.0, value=0.5, step=0.000001)
    dose = st.number_input('Enter the value for dose (Be sure to confirm the units for dose match those from the best fitting dose response model.)', min_value=0.0, max_value=100000.0, value=10.0, step=0.000001)

    if st.button('Calculate Risk for Exponential'):
        # Calculate risk
        risk = calculate_risk_exp(k, dose)

        # Display risk
        st.write(f'The calculated risk for exponential distribution is: {risk:.4f}')

elif distributions == 'Beta-Poisson':
    
    
    beta_poisson_options = st.select_slider(
        'Choose type of Beta-Poisson',
        options = ['Regular', f'Approximate(N_50)']
    )
    
    # Use sliders to select parameters for beta-poisson distribution
    if beta_poisson_options == 'Regular':
        regular = st.latex("1 - [1 + \\frac{dose}{\\beta}]^{-\\alpha}")
        alpha = st.number_input('Enter the value for alpha', min_value=0.0, max_value=10.0, value=0.0, step=0.00000001)
        beta = st.number_input('Enter the value for beta', min_value=0.0, max_value=10.0, value=0.0, step=0.00000001)
        dose = st.number_input('Enter the value for dose', min_value=0.0, max_value=100.0, value=0.0, step=0.00000001)
        if st.button('Calculate Risk for Beta-Poisson'):
        # Calculate risk
            risk = calculate_risk_beta_poisson_regular(alpha, beta, dose)

        # Display risk
            st.write(f'The calculated risk for beta-poisson distribution is: {risk:.4f}')
    else:
        approx_beta = st.latex("1 - [1 + dose \\times \\frac{(2^{\\frac{1}{\\alpha}} - 1)}{N_{50}}]^{-\\alpha}")
        n_or_beta = st.select_slider(
            'Enter beta or N_50',
            options = ['Beta', 'N_50']

        )
        if n_or_beta == 'Beta':
            alpha = st.number_input('Enter the value for alpha', min_value=0.0, max_value=10.0, value=1.0, step=0.00000001)
            
            beta = st.number_input('Enter the value for beta', min_value=0.0, max_value=10.0, value=1.0, step=0.00000001)
            dose = st.number_input('Enter the value for dose', min_value=0.0, max_value=100.0, value=10.0, step=0.00000001)
            if st.button('Calculate Risk for Beta-Poisson'):
            # Calculate risk
                risk = calculate_risk_beta_poisson_approximate(dose, alpha, beta, False)

            # Display risk
                st.write(f'The calculated risk for beta-poisson distribution is: {risk:.4f}')
        else:
            alpha = st.number_input('Enter the value for alpha', min_value=0.0, max_value=10.0, value=1.0, step=0.00000001)
            n_50 = st.number_input(f'Enter the value for N50', min_value=0.0, max_value=10.0, value=1.0, step=0.00000001)
            dose = st.number_input('Enter the value for dose', min_value=0.0, max_value=100.0, value=10.0, step=0.00000001)
            if st.button('Calculate Risk for Beta-Poisson'):
            # Calculate risk
                risk = calculate_risk_beta_poisson_approximate(dose, alpha, n_50, True)

            # Display risk
                st.write(f'The calculated risk for beta-poisson distribution is: {risk:.4f}')

    
