import streamlit as st
import numpy as np

st.markdown("""
<style>
.big-font {
    color: blue;
    font-size:50px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">Exponential Distribution Risk Calculator</p>', unsafe_allow_html=True)

st.latex('Risk = 1 - exp(-k * dose) \\\ Probability = 1 - (1- Risk)^{\\alpha}')



# Define a function to calculate risk
def calculate_risk(k, dose, alpha):
    risk = 1 - np.exp(-k * dose)
    prob = 1-np.power((1-risk),alpha)
    return prob


k = st.number_input('Enter the value for k')
dose = st.number_input('Enter the value for dose')
alpha = st.number_input('Enter alpha')
if st.button('Calculate Risk'):
    
    risk = calculate_risk(k, dose,alpha)

    
    st.write(f'The calculated risk is: {risk:.4f}')
