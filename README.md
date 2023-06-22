# qmra-riskapp


## Description

This app enables users to calculate risk of infection given the pathogen and dose. Purpose of this app is to explore different pathogens and their risks of infection.

Application runs on Streamlit and pulls all data from a MongoDB server. 

## Installation 

In order to install all the required packages for the application when inside the root directory of the project run: \
```pip install -r requirements.txt ```

You need to create a .streamlit/secrets.toml file. That file is used to load the secrets to connect to the MongoDB server. The file should look like this:
```
[mongo]
host = "<address of the MongoDB server>"
port = <PORT usually 27017>
```

##  Running the application 

``` streamlit run main.py ```


