import plotly.express as px
import collections.abc
import streamlit as st
import numpy as np
import random 
import string
import requests
from calc import calculate_risk_exp, calculate_risk_beta_poisson_regular, calculate_risk_beta_poisson_approximate
import uuid
import json
import  streamlit_toggle as tog
import matplotlib.pyplot as plt


st.set_page_config(page_title="Distribution Calculator")

st.title('Pathogen Distribution Calculator')
st.divider() 

@st.cache_data
def get_pathogens():
    
    response = requests.get('https://qmrawiki.org/data/pathogens.json')
    data = json.loads(response.content)['pathogens']
    pathogen_dict = {}
    for pathogen in data:
        name = data[pathogen]['pathogen_name']
        pathogen_dict[name] = {}
        for path in data[pathogen]:
            if path != 'pathogen_name':
                pathogen_dict[name][path] = data[pathogen][path]

    
    return pathogen_dict

@st.cache_data
def get_pathogen_names(pathogen_dict):
    pathogen__names_list = [pathogen for pathogen in pathogen_dict]
    return tuple(pathogen__names_list)

def display_microbial_group(data, pathogen):
    st.write(f'Microbial group: {data[pathogen]["microbial_group"]}')


    
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
                        key=f"for_plot_button_{identifier}_{identifier*54}", 
                        default_value=False, 
                        label_after = True, 
                        inactive_color = '#D3D3D3', 
                        active_color="#11567f", 
                        track_color="#29B5E8"
                        )

def calc_plot_dist(epsilon=1e-6, n_steps_after_convergence=4, **kwargs):
    """
    FUNCTION TO CALCULATE THE VALUES FOR THE PLOT BASED ON THE DIST
    """
    dose_range = np.logspace(-2, 300000, num=100000, base=10.0)
    risk_range = []

    converge_count = 0  # count steps after convergence
    last_risk = None  # keep track of the last computed risk

    for dose in dose_range:
        if kwargs['dist'] == 'exp':
            risk = calculate_risk_exp(kwargs['k'], dose, 0, False)
        elif kwargs['dist'] == 'beta-approx':
            risk = calculate_risk_beta_poisson_approximate(dose, kwargs['alpha'], kwargs['param'], True)
        elif kwargs['dist'] == 'beta-reg':
            risk = calculate_risk_beta_poisson_regular(kwargs['alpha'], kwargs['beta'], dose)

        # if we've seen at least one risk before
        if last_risk is not None:
            # if the difference between this risk and the last is less than epsilon
            if abs(risk - last_risk) < epsilon:
                # increment the convergence count
                converge_count += 1
                # if we've counted enough steps past convergence
                if converge_count >= n_steps_after_convergence:
                    # stop computing risks
                    break
            else:
                # if the risks haven't converged, reset the counter
                converge_count = 0

        risk_range.append(risk)
        last_risk = risk  # update the last computed risk

    return [dose_range[:len(risk_range)], risk_range]


def session_state_loader(**kwargs):


    if not st.session_state:
        for arg in kwargs:
            st.session_state[kwargs[arg][0]] = kwargs[arg][1]
    else:
        for arg in kwargs:
            if kwargs[arg][0] not in st.session_state:
                st.session_state[kwargs[arg][0]] = kwargs[arg][1]


left_column, right_column = st.columns(2)


def display_exponential(data, identifier, pathogen = 'None', k_optimal=0.0, is_optimal = False, for_dose = False):
    
    if is_optimal == False: 
        st.subheader('Exponential Distribution')
    else:
        st.subheader(f'Exponential Distribution - {pathogen}')
        display_microbial_group(data, pathogen)
    st.latex("1 - exp(-k \\times dose)")

    k_key = (f'k_exp_{identifier}', 0.1)
    dose_key = (f'dose_exp_{identifier}', 0.1)
    risk_key = (f'risk_exp_{identifier}', 0.1)
    k_key_optimal = (f"{k_key}_optimal", float(k_optimal))

    for_dose = for_dose_button(identifier)
    session_state_loader(k_key = k_key, dose_key = dose_key, risk_key = risk_key, k_key_optimal = k_key_optimal)


    if for_dose == False:
        plot_dist_ = plot_dist_button(identifier)
        
        k_optimal = float(k_optimal)

        if is_optimal:
            
            k = st.number_input('Enter the value for k', key=k_key_optimal[0], min_value=0.0, max_value=1.0, value=k_optimal, step=0.000001, format="%.5f")
            dose = st.number_input('Enter the value for dose', key=dose_key[0], min_value=0.0, max_value=100000.0, value=st.session_state[dose_key[0]], step=0.000001, format="%.5f")
        else:
            k = st.number_input('Enter the value for k', key=k_key[0], min_value=0.0, max_value=1.0, value=st.session_state[k_key[0]], step=0.000001, format="%.5f")
            dose = st.number_input('Enter the value for dose', key=dose_key[0], min_value=0.0, max_value=1000.0, value=st.session_state[dose_key[0]], step=0.000001, format="%.5f")
        
        risk = calculate_risk_exp(float(k), dose)
        st.write(f'The calculated risk for exponential distribution is: {risk:.4f}')
        
        if plot_dist_:
            doses, risks = calc_plot_dist(dist = 'exp', k = k)
            
            fig, ax = plt.subplots()
            ax.plot(doses,risks)
            ax.set_xscale('log')
            st.pyplot(fig)
    else:
        
        if is_optimal:
            risk = st.number_input('Enter the value for risk', key=risk_key[0], min_value=0.0, max_value=1.0, value=0.0, step = 0.0001)
            dose = calculate_risk_exp(k_optimal, 0, risk, for_dose=True)
            st.write(f"The dose calculated for risk and optimal parameters is: {dose:.4f}")
        else:
            
            k = st.number_input('Enter the value for k', key=k_key[0], min_value=0.0, max_value=1.0, value=st.session_state[k_key], step=0.000001)
            risk = st.number_input('Enter the value for risk', key=risk_key[0], min_value=0.0, max_value=1.0, value=st.session_state[risk_key[0]], step=0.000001)
            dose = calculate_risk_exp(k, 0, risk, True)
            st.write(f"The dose calculated for risk and optimal parameters is: {dose:.4f}")



def display_beta_poisson_regular(data, identifier, pathogen = 'None', alpha_optimal=0.1, beta_optimal=0.1, is_optimal=False):
    
    
    if is_optimal == False: 
        st.subheader('Beta-Poisson-Regular')
    else:
        st.subheader(f'Beta-Poisson-Regular - {pathogen}') 
        display_microbial_group(data, pathogen)
    st.latex("1 - [1 + \\frac{dose}{\\beta}]^{-\\alpha}")

    alpha_key = (f'alpha_beta_{identifier}', 0.1)
    alpha_key_optimal = (f'alpha_beta_{identifier}_optimal', float(alpha_optimal))
    beta_key = (f'beta_beta_{identifier}', 0.1)
    beta_key_optimal = (f'beta_beta_{identifier}_optimal', beta_optimal)
    dose_key = (f'dose_beta_{identifier}',0.1)
    risk_key = (f'risk_beta_regular_{identifier}',0.1)

    session_state_loader(alpha_key=alpha_key, beta_key=beta_key, dose_key=dose_key, risk_key=risk_key, alpha_key_optimal=alpha_key_optimal,
                beta_key_optimal=beta_key_optimal)


    for_dose = for_dose_button(identifier)
    plot_dist_ = plot_dist_button(identifier)

    if for_dose == False:
        
        if is_optimal:
            alpha = st.number_input('Enter the value for alpha', key=alpha_key_optimal[0], min_value=0.000001, max_value=10.0, value=alpha_optimal, step=0.00000001)
            beta = st.number_input('Enter the value for beta', key=beta_key_optimal[0], min_value=0.0, max_value=1000000.0, value=beta_optimal, step=0.00000001)
            dose = st.number_input('Enter the dose value', key=dose_key[0], min_value=0.0, max_value=1000000.0, value=0.0, step=0.00000001)
        else:
            alpha = st.number_input('Enter the value for alpha', key=alpha_key[0], min_value=0.000001, max_value=10.0, value=st.session_state[alpha_key[0]], step=0.00000001)
            beta = st.number_input('Enter the value for beta', key=beta_key[0], min_value=0.000001, max_value=1000000.0, value=st.session_state[beta_key[0]], step=0.00000001)
            dose = st.number_input('Enter the dose value', key=dose_key[0], min_value=0.0, max_value=10000000.0, value=st.session_state[dose_key[0]], step=0.00000001)
    
        risk = calculate_risk_beta_poisson_regular(alpha, beta, dose)
        st.write(f'The calculated risk for beta-poisson regular distribution is: {risk:.4f}')
    else:
        
        if is_optimal:
            alpha = st.number_input('Enter the value for alpha', key=alpha_key_optimal[0], min_value=0.0, max_value=10.0, value=alpha_optimal, step=0.00000001)
            beta = st.number_input('Enter the value for beta', key=beta_key_optimal[0], min_value=0.0, max_value=1000000.0, value=beta_optimal, step=0.00000001)
            risk = st.number_input('Enter the value for risk', key=risk_key[0], min_value=0.0, max_value=1.0, value=st.session_state[risk_key[0]], step=0.00000001)
        else:
            alpha = st.number_input('Enter the value for alpha', key=alpha_key[0], min_value=0.0, max_value=10.0, value=st.session_state[alpha_key[0]], step=0.00000001)
            beta = st.number_input('Enter the value for beta', key=beta_key[0], min_value=0.0, max_value=1000000.0, value=st.session_state[beta_key[0]], step=0.00000001)
            risk = st.number_input('Enter the value for risk', key=risk_key[0], min_value=0.0, max_value=1.0, value=st.session_state[risk_key[0]], step=0.00000001)
        dose = calculate_risk_beta_poisson_regular(alpha, beta, 0.0, risk, True)
        st.write(f'The calculated dose for risk of {risk:.4f} is {dose:.4f}')
        if plot_dist_:
            doses, risks = calc_plot_dist(dist = 'beta-reg', alpha = alpha, beta=beta)
            
            fig, ax = plt.subplots()
            ax.plot(doses,risks)
            ax.set_xscale('log')
            st.pyplot(fig)



def display_beta_poisson_approximate_beta(identifier):
    st.subheader('Beta-Poisson Distribution - Approximate (Beta)')
    st.latex("N_{50} = \\beta * [2^{\\frac{1}{\\alpha}} - 1]")
    st.latex("1 - [1 + dose \\times \\frac{(2^{\\frac{1}{\\alpha}} - 1)}{N_{50}}]^{-\\alpha}")

    alpha_key = (f'alpha_beta_approx_{identifier}',0.1)
    beta_key = (f'beta_beta_approx_{identifier}',0.1)
    dose_key = (f'dose_beta_approx_{identifier}',1.0)

    session_state_loader(alpha_key=alpha_key, beta_key=beta_key, dose_key=dose_key)
    
    alpha = st.number_input('Enter the value for alpha', key=alpha_key[0], min_value=0.0, max_value=10.0, value=st.session_state[alpha_key[0]], step=0.00000001)
    beta = st.number_input('Enter the value for beta', key=beta_key[0], min_value=0.0, max_value=10.0, value=st.session_state[beta_key[0]], step=0.00000001)
    dose = st.number_input('Enter the dose value', key=dose_key[0], min_value=0.0, max_value=100.0, value=st.session_state[dose_key[0]], step=0.00000001)

    risk = calculate_risk_beta_poisson_approximate(dose, alpha, beta, False)
    st.write(f'The calculated risk for beta-poisson approximate (Beta) distribution is: {risk:.4f}')


def display_beta_poisson_approximate_n50(data, identifier, pathogen = 'None', alpha_optimal=0.0, n50_optimal=0.0, is_optimal=False):

    if is_optimal == False: 
        st.subheader('Beta-Poisson-Approximate')
    else:
        st.subheader(f'Beta-Poisson-Approximate - {pathogen}')
        display_microbial_group(data, pathogen)
    st.latex("1 - [1 + dose \\times \\frac{(2^{\\frac{1}{\\alpha}} - 1)}{N_{50}}]^{-\\alpha}")
    for_dose = for_dose_button(identifier)
    plot_dist_ = plot_dist_button(identifier)
    try:
        n50_optimal = float(n50_optimal)
    except:
        n50_optimal = float(n50_optimal.split(' ')[0])
            
    alpha_key = (f'alpha_n50_approx_{identifier}',0.1)
    alpha_key_optimal = (f'alpha_n50_approx_{identifier}_optimal', float(alpha_optimal))
    n50_key = (f'n50_n50_approx_{identifier}', 0.1)
    n50_key_optimal = (f'n50_n50_approx_{identifier}_optimal', n50_optimal)
    dose_key = (f'dose_n50_approx_{identifier}', 1.0)
    risk_key = (f'risk_n50_approx_{identifier}', 0.1)

    session_state_loader(alpha_key=alpha_key, n50_key=n50_key, dose_key=dose_key, risk_key=risk_key, n50_key_optimal = n50_key_optimal, 
              alpha_key_optimal = alpha_key_optimal)

    if for_dose == False:
        
        if is_optimal:
            session_state_loader(alpha_key=alpha_key, n50_key=n50_key, dose_key=dose_key, risk_key=risk_key, n50_key_optimal = n50_key_optimal, 
                alpha_key_optimal = alpha_key_optimal)
            try:
                n50_optimal = float(n50_optimal)
            except:
                n50_optimal = float(n50_optimal.split(' ')[0])
            alpha_optimal = float(alpha_optimal)
            alpha = st.number_input('Enter the value for alpha',key=alpha_key_optimal[0],  min_value=0.000001, max_value=10.0, value=alpha_optimal, step=0.00000001)
            n_50 = st.number_input('Enter the value for N50', key=n50_key_optimal[0], min_value=0.00000001, max_value=1000000000.0, value=n50_optimal, step=0.00000001)
            dose = st.number_input('Enter the dose value', key=dose_key[0], min_value=0.0, max_value=100000000.0, value=0.0, step=0.00000001)

        else:
            alpha = st.number_input('Enter the value for alpha', key=alpha_key[0], min_value=0.00000001, max_value=10.0, value=st.session_state[alpha_key[0]], step=0.00000001)
            n_50 = st.number_input('Enter the value for N50', key=n50_key[0], min_value=0.000000001, max_value=10.0, value=st.session_state[n50_key[0]], step=0.00000001)
            dose = st.number_input('Enter the dose value', key=dose_key[0], min_value=0.0, max_value=100.0, value=st.session_state[dose_key[0]], step=0.00000001)

        risk = calculate_risk_beta_poisson_approximate(dose, alpha, n_50, True, 0, False)
        st.write(f'The calculated risk for beta-poisson approximate (N_50) distribution is: {risk:.4f}')
        if plot_dist_:
            doses, risks = calc_plot_dist(dist = 'beta-approx', alpha = alpha, param=n_50)
            
            fig, ax = plt.subplots()
            ax.plot(doses,risks)
            ax.set_xscale('log')
            st.pyplot(fig)
    else:
        
        if is_optimal:
            alpha = st.number_input('Enter the value for alpha', key=alpha_key_optimal[0], min_value=0.0, max_value=10.0, value=alpha_optimal, step=0.00000001)
            beta = st.number_input('Enter the value for N50', key=n50_key_optimal[0], min_value=0.0, max_value=1000000.0, value=n50_optimal, step=0.00000001)
            risk = st.number_input('Enter the value for risk', key=risk_key[0], min_value=0.0, max_value=1.0, value=st.session_state[risk_key[0]], step=0.00000001)
        else:
            alpha = st.number_input('Enter the value for alpha', key=alpha_key[0], min_value=0.00000001, max_value=10.0, value=st.session_state[alpha_key[0]], step=0.00000001)
            beta = st.number_input('Enter the value for N50', key=n50_key[0], min_value=0.00000001, max_value=1000000.0, value=st.session_state[n50_key[0]], step=0.00000001)
            risk = st.number_input('Enter the value for risk', key=risk_key[0], min_value=0.000001, max_value=1.0, value=st.session_state[risk_key[0]], step=0.00000001)
        dose = calculate_risk_beta_poisson_approximate(0.0, alpha, beta, True, risk, True)
        st.write(f'The calculated dose for risk of {risk:.4f} is {dose:.4f}')
        
        
        

def display_selection(key, data, pathogen_names_list):
    
    #URL PARSING
    if "pathogen" in st.experimental_get_query_params():
        initial_selection = st.selectbox("Choose a Distribution or a Pathogen",
            ('Distribution', 'Pathogen'), key=key+'1234_selection', index = 1)
    else:
        initial_selection = st.selectbox("Choose a Distribution or a Pathogen",
            ('Distribution', 'Pathogen'), key=key+'123')
    if initial_selection == 'Distribution':

        selection = st.selectbox("Choose a Distribution",
            ("Exponential Distribution", "Beta-Poisson Distribution - Regular", "Beta-Poisson Distribution - Approximate (N_50)"),
            key=key)
        if selection == "Exponential Distribution":
            display_exponential(data, key,0.0,False)
        elif selection == "Beta-Poisson Distribution - Regular":
            display_beta_poisson_regular(data, key, 0.0,0.0, False)
        elif selection == "Beta-Poisson Distribution - Approximate (Beta)":
            display_beta_poisson_approximate_beta(key)
        elif selection == "Beta-Poisson Distribution - Approximate (N_50)":
            display_beta_poisson_approximate_n50(data, key, 0.0, 0.0, False)
    else:
        
        
        #URL PARSING
         
        selection_pathogen = st.selectbox('Chose a Pathogen',
        np.array(pathogen_names_list),
        key=f"nonpathogen_{key}")
        
        if 'k' in data[selection_pathogen]:
            display_exponential(data, key+'optimal', selection_pathogen, data[selection_pathogen]['k'], True)
        elif 'beta' in data[selection_pathogen]:
            display_beta_poisson_regular(data, key+'optimal', selection_pathogen, data[selection_pathogen]['a'], data[selection_pathogen]['beta'], True)
        elif 'n50' in data[selection_pathogen]:
            display_beta_poisson_approximate_n50(data, key+'optimal', selection_pathogen, data[selection_pathogen]['a'], data[selection_pathogen]['n50'], True)



def main():
    
    
    data = get_pathogens()
    pathogen_names = get_pathogen_names(data)
    
    with left_column:
        st.header("Box 1") 
        display_selection("select_1", data, pathogen_names)    
        st.divider() 
        
    with right_column:
        st.header("Box 2")
        display_selection("select_2", data, pathogen_names)
        st.divider() 

if __name__ == "__main__":
    main()


    
