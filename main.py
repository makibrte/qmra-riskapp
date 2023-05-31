import plotly.express as px
import streamlit as st
import numpy as np
import random 
import string
from calc import calculate_risk_exp, calculate_risk_beta_poisson_regular, calculate_risk_beta_poisson_approximate
import uuid
from scipy.stats import betaprime
import  streamlit_toggle as tog
import matplotlib.pyplot as plt

st.set_page_config(page_title="Distribution Calculator")

st.title('Pathogen Distribution Calculator')
st.divider() 

optimal_parameters = {'Ebola': {'N50' : 2.167, 'alpha': 1.23},
                    'Salmonella' : {'N50' : 23600.0, 'alpha' : 0.3126},
                    'Campylobacter jejuni' : {'N50': 896.0, 'alpha':0.145},
                    'Cryptosporidium parvum' : {'k' : 0.0042},
                    'Giardia lamblia' : {'k':0.02},
                    'Norovirus' : {'N50':1845493.0 , 'alpha':0.04, 'beta' : 0.055},
                    'Rotavirus' : {'N50':6.17, 'alpha':0.25, 'beta':0.42},
                    'Echo' : {'N50':1052.0, 'alpha':0.401, 'beta':227.2},
                    'Salm': {'N50':1003.0, 'alpha':0.33, 'beta':139.9}}

def for_dose_button(identifier):
    return tog.st_toggle_switch(label="Calculate for Dose", 
                        key=f"for_dose_button_{identifier}", 
                        default_value=False, 
                        label_after = True, 
                        inactive_color = '#D3D3D3', 
                        active_color="#11567f", 
                        track_color="#29B5E8"
                        )
                        
def plot_dist_button(identifier):
    return tog.st_toggle_switch(label="Plot the Distribution", 
                        key=f"for_plot_button_{identifier}", 
                        default_value=False, 
                        label_after = True, 
                        inactive_color = '#D3D3D3', 
                        active_color="#11567f", 
                        track_color="#29B5E8"
                        )

def plot_dist(**kwargs):
    """
    FUNCTION TO CALCULATE THE VALUES FOR THE PLOT BASED ON THE DIST
    """
    dose_range = np.logspace(0, 7, num=1000, base=10.0)
    if kwargs['dist'] == 'exp':
        risk_range = [calculate_risk_exp(kwargs['k'], dose, 0, False) for dose in dose_range]

    elif kwargs['dist'] == 'beta-approx':
        risk_range = [calculate_risk_beta_poisson_approximate(dose, kwargs['alpha'], kwargs['param'], True) for dose in dose_range]

    elif kwargs['dist'] == 'beta-reg':
        risk_range = [calculate_risk_beta_poisson_regular(kwargs['alpha'], kwargs['beta'], dose) for dose in dose_range]
    return [dose_range, risk_range]

left_column, right_column = st.columns(2)


def display_exponential(identifier, k_optimal, is_optimal = False, for_dose = False):
    st.subheader('Exponential Distribution')
    st.latex("1 - exp(-k \\times dose)")

    k_key = f'k_exp_{identifier}'
    dose_key = f'dose_exp_{identifier}'
    risk_key = f'risk_exp_{identifier}'

    for_dose = for_dose_button(identifier)
    if not st.session_state:
        st.session_state[k_key] = 0.5
        st.session_state[dose_key] = 10.0
        st.session_state[risk_key] = 0.0

            
    elif k_key not in st.session_state or dose_key not in st.session_state:
        st.session_state[k_key] = 0.5
        st.session_state[dose_key] = 10.0
        st.session_state[risk_key] = 0.0


    if for_dose == False:
        plot_dist_ = plot_dist_button(identifier)
        
        

        if is_optimal:
            k = st.number_input('Enter the value for k', key=f"{k_key}_optimal", min_value=0.0, max_value=1.0, value=k_optimal, step=0.000001, format="%.5f")
            dose = st.number_input('Enter the value for dose', key=dose_key, min_value=0.0, max_value=100000.0, value=st.session_state[dose_key], step=0.000001, format="%.5f")
        else:
            k = st.number_input('Enter the value for k', key=k_key, min_value=0.0, max_value=1.0, value=st.session_state[k_key], step=0.000001, format="%.5f")
            dose = st.number_input('Enter the value for dose', key=dose_key, min_value=0.0, max_value=1000.0, value=st.session_state[dose_key], step=0.000001, format="%.5f")
        
        risk = calculate_risk_exp(k, dose)
        st.write(f'The calculated risk for exponential distribution is: {risk:.4f}')
        if plot_dist_:
            doses, risks = plot_dist(dist = 'exp', k = k)
            
            fig, ax = plt.subplots()
            ax.plot(doses,risks)
            ax.set_xscale('log')
            st.pyplot(fig)
    else:
        if not st.session_state:
            st.session_state[k_key] = 0.5
            st.session_state[dose_key] = 10.0
            st.session_state[risk_key] = 0.0

            
        elif k_key not in st.session_state or dose_key not in st.session_state:
            st.session_state[k_key] = 0.5
            st.session_state[dose_key] = 10.0
            st.session_state[risk_key] = 0.0
        if is_optimal:
            risk = st.number_input('Enter the value for risk', key=risk_key, min_value=0.0, max_value=1.0, value=0.0, step = 0.0001)
            dose = calculate_risk_exp(k_optimal, 0, risk, for_dose=True)
            st.write(f"The dose calculated for risk and optimal parameters is: {dose:.4f}")
        else:
            
            k = st.number_input('Enter the value for k', key=k_key, min_value=0.0, max_value=1.0, value=st.session_state[k_key], step=0.000001)
            risk = st.number_input('Enter the value for risk', key=risk_key, min_value=0.0, max_value=1.0, value=st.session_state[risk_key], step=0.000001)
            dose = calculate_risk_exp(k, 0, risk, True)
            st.write(f"The dose calculated for risk and optimal parameters is: {dose:.4f}")



def display_beta_poisson_regular(identifier, alpha_optimal, beta_optimal, is_optimal=False):
    st.subheader('Beta-Poisson Distribution - Regular')
    st.latex("1 - [1 + \\frac{dose}{\\beta}]^{-\\alpha}")

    alpha_key = f'alpha_beta_{identifier}'
    beta_key = f'beta_beta_{identifier}'
    dose_key = f'dose_beta_{identifier}'
    risk_key = f'risk_beta_regular_{identifier}'

    for_dose = for_dose_button(identifier)
    plot_dist_ = plot_dist_button(identifier)
    if for_dose == False:
        if is_optimal == False:
            if not st.session_state:
                st.session_state[alpha_key] = 1.0
                st.session_state[beta_key] = 1.0
                st.session_state[dose_key] = 10.0

            
            elif alpha_key not in st.session_state or beta_key not in st.session_state or dose_key not in st.session_state:
                st.session_state[alpha_key] = 1.0
                st.session_state[beta_key] = 1.0
                st.session_state[dose_key] = 10.0
        if is_optimal:
            alpha = st.number_input('Enter the value for alpha', key=alpha_key, min_value=0.0, max_value=10.0, value=alpha_optimal, step=0.00000001)
            beta = st.number_input('Enter the value for beta', key=beta_key, min_value=0.0, max_value=1000000.0, value=beta_optimal, step=0.00000001)
            dose = st.number_input('Enter the dose value', key=dose_key, min_value=0.0, max_value=1000000.0, value=0.0, step=0.00000001)
        else:
            alpha = st.number_input('Enter the value for alpha', key=alpha_key, min_value=0.0, max_value=10.0, value=st.session_state[alpha_key], step=0.00000001)
            beta = st.number_input('Enter the value for beta', key=beta_key, min_value=0.0, max_value=1000000.0, value=st.session_state[beta_key], step=0.00000001)
            dose = st.number_input('Enter the dose value', key=dose_key, min_value=0.0, max_value=10000000.0, value=st.session_state[dose_key], step=0.00000001)
    
        risk = calculate_risk_beta_poisson_regular(alpha, beta, dose)
        st.write(f'The calculated risk for beta-poisson regular distribution is: {risk:.4f}')
    else:
        if not st.session_state:
                st.session_state[alpha_key] = 1.0
                st.session_state[beta_key] = 1.0
                st.session_state[dose_key] = 10.0
                st.session_state[risk_key] = 0.0

            
        elif alpha_key not in st.session_state or beta_key not in st.session_state or dose_key not in st.session_state:
            st.session_state[alpha_key] = 1.0
            st.session_state[beta_key] = 1.0
            st.session_state[dose_key] = 10.0
            st.session_state[risk_key] = 0.0
        if is_optimal:
            alpha = st.number_input('Enter the value for alpha', key=alpha_key, min_value=0.0, max_value=10.0, value=alpha_optimal, step=0.00000001)
            beta = st.number_input('Enter the value for beta', key=beta_key, min_value=0.0, max_value=1000000.0, value=beta_optimal, step=0.00000001)
            risk = st.number_input('Enter the value for risk', key=risk_key, min_value=0.0, max_value=1.0, value=st.session_state[risk_key], step=0.00000001)
        else:
            alpha = st.number_input('Enter the value for alpha', key=alpha_key, min_value=0.0, max_value=10.0, value=st.session_state[alpha_key], step=0.00000001)
            beta = st.number_input('Enter the value for beta', key=beta_key, min_value=0.0, max_value=1000000.0, value=st.session_state[beta_key], step=0.00000001)
            risk = st.number_input('Enter the value for risk', key=risk_key, min_value=0.0, max_value=1.0, value=st.session_state[risk_key], step=0.00000001)
        dose = calculate_risk_beta_poisson_regular(alpha, beta, 0.0, risk, True)
        st.write(f'The calculated dose for risk of {risk:.4f} is {dose:.4f}')
        if plot_dist_:
            doses, risks = plot_dist(dist = 'beta-reg', alpha = alpha, beta=beta)
            
            fig, ax = plt.subplots()
            ax.plot(doses,risks)
            ax.set_xscale('log')
            st.pyplot(fig)



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







def display_beta_poisson_approximate_n50(identifier, alpha_optimal, n50_optimal, is_optimal=False):
    st.subheader('Beta-Poisson Distribution - Approximate (N_50)')
    st.latex("1 - [1 + dose \\times \\frac{(2^{\\frac{1}{\\alpha}} - 1)}{N_{50}}]^{-\\alpha}")
    for_dose = for_dose_button(identifier)
    plot_dist_ = plot_dist_button(identifier)

    alpha_key = f'alpha_n50_approx_{identifier}'
    n50_key = f'n50_n50_approx_{identifier}'
    dose_key = f'dose_n50_approx_{identifier}'
    risk_key = f'risk_n50_approx_{identifier}'
    if for_dose == False:
        if is_optimal == False:

            if not st.session_state:
                st.session_state[alpha_key] = 1.0
                st.session_state[n50_key] = 1.0
                st.session_state[dose_key] = 10.0
            elif alpha_key not in st.session_state or n50_key not in st.session_state or dose_key not in st.session_state:
                st.session_state[alpha_key] = 1.0
                st.session_state[n50_key] = 1.0
                st.session_state[dose_key] = 10.0
        if is_optimal:

            alpha = st.number_input('Enter the value for alpha', key=alpha_key+'optimal', min_value=0.0, max_value=10.0, value=alpha_optimal, step=0.00000001)
            n_50 = st.number_input('Enter the value for N50', key=n50_key, min_value=0.0, max_value=1000000.0, value=n50_optimal, step=0.00000001)
            dose = st.number_input('Enter the dose value', key=dose_key, min_value=0.0, max_value=100000000.0, value=0.0, step=0.00000001)

        else:
            alpha = st.number_input('Enter the value for alpha', key=alpha_key, min_value=0.0, max_value=10.0, value=st.session_state[alpha_key], step=0.00000001)
            n_50 = st.number_input('Enter the value for N50', key=n50_key, min_value=0.0, max_value=10.0, value=st.session_state[n50_key], step=0.00000001)
            dose = st.number_input('Enter the dose value', key=dose_key, min_value=0.0, max_value=100.0, value=st.session_state[dose_key], step=0.00000001)

        risk = calculate_risk_beta_poisson_approximate(dose, alpha, n_50, True, 0, False)
        st.write(f'The calculated risk for beta-poisson approximate (N_50) distribution is: {risk:.4f}')
        if plot_dist_:
            doses, risks = plot_dist(dist = 'beta-approx', alpha = alpha, param=n_50)
            
            fig, ax = plt.subplots()
            ax.plot(doses,risks)
            ax.set_xscale('log')
            st.pyplot(fig)
    else:
        if not st.session_state:
                st.session_state[alpha_key] = 1.0
                st.session_state[dose_key] = 10.0
                st.session_state[risk_key] = 0.0

            
        elif alpha_key not in st.session_state or n50_key not in st.session_state or dose_key not in st.session_state or risk_key not in st.session_state:
            st.session_state[alpha_key] = 1.0
            st.session_state[dose_key] = 10.0
            st.session_state[risk_key] = 0.0
        if is_optimal:
            alpha = st.number_input('Enter the value for alpha', key=alpha_key, min_value=0.0, max_value=10.0, value=alpha_optimal, step=0.00000001)
            beta = st.number_input('Enter the value for N50', key=n50_key, min_value=0.0, max_value=1000000.0, value=n50_optimal, step=0.00000001)
            risk = st.number_input('Enter the value for risk', key=risk_key, min_value=0.0, max_value=1.0, value=st.session_state[risk_key], step=0.00000001)
        else:
            alpha = st.number_input('Enter the value for alpha', key=alpha_key, min_value=0.0, max_value=10.0, value=st.session_state[alpha_key], step=0.00000001)
            beta = st.number_input('Enter the value for N50', key=n50_key, min_value=0.0, max_value=1000000.0, value=st.session_state[n50_key], step=0.00000001)
            risk = st.number_input('Enter the value for risk', key=risk_key, min_value=0.0, max_value=1.0, value=st.session_state[risk_key], step=0.00000001)
        dose = calculate_risk_beta_poisson_approximate(0.0, alpha, beta, True, risk, True)
        st.write(f'The calculated dose for risk of {risk:.4f} is {dose:.4f}')
        
        
        

def display_selection(key):
    initial_selection = st.selectbox("Choose a Distribution or a Pathogen",
        ('Distribution', 'Pathogen'), key=key+'123')
    if initial_selection == 'Distribution':

        selection = st.selectbox("Choose a Distribution",
            ("Exponential Distribution", "Beta-Poisson Distribution - Regular", "Beta-Poisson Distribution - Approximate (Beta)",
            "Beta-Poisson Distribution - Approximate (N_50)"),
            key=key)
        if selection == "Exponential Distribution":
            display_exponential(key,0.0,False)
        elif selection == "Beta-Poisson Distribution - Regular":
            display_beta_poisson_regular(key, 0.0,0.0, False)
        elif selection == "Beta-Poisson Distribution - Approximate (Beta)":
            display_beta_poisson_approximate_beta(key)
        elif selection == "Beta-Poisson Distribution - Approximate (N_50)":
            display_beta_poisson_approximate_n50(key, 0.0, 0.0, False)
    else:
        #TODO: FIGURE OUT THE LAST FEW PATHOGENS
        selection_pathogen = st.selectbox('Chose a Pathogen',
            ('Ebola', 'Salmonella', 'Campylobacter jejuni', 'Cryptosporidium parvum', 'Giardia lamblia',
            'Norovirus', 'Rotavirus', 'Echo', 'Salm'))
        if 'k' in optimal_parameters[selection_pathogen]:
            display_exponential(key, optimal_parameters[selection_pathogen]['k'], True)
        elif 'beta' in optimal_parameters[selection_pathogen]:
            display_beta_poisson_regular(key, optimal_parameters[selection_pathogen]['alpha'], optimal_parameters[selection_pathogen]['beta'], True)
        elif 'N50' in optimal_parameters[selection_pathogen]:
            display_beta_poisson_approximate_n50(key, optimal_parameters[selection_pathogen]['alpha'], optimal_parameters[selection_pathogen]['N50'], True)


with left_column:
    st.header("Box 1")
    display_selection(key="select_1")    
    st.divider() 
        


with right_column:
    st.header("Box 2")
    display_selection(key="select_2")
    st.divider() 


        
"""
REMOVE THE QUOTATIONS TO DISPLAY BOTTOM ROW


left_column_2, right_column_2 = st.columns(2)
with left_column_2:
    st.header("Box 3")
    display_selection(key="select_3")


with right_column_2:
    st.header("Box 4")
    display_selection(key="select_4")

    """
