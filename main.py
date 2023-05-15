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


# Define a function to calculate risk
def calculate_risk(k, dose):
    risk = 1 - np.exp(-k * dose)
    return risk


k = st.number_input('Enter the value for k')
dose = st.number_input('Enter the value for dose')

if st.button('Calculate Risk'):
    
    risk = calculate_risk(k, dose)

    
    st.write(f'The calculated risk is: {risk:.4f}')
