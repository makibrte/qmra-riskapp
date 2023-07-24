# qmra-riskapp


## Description

This app enables users to calculate risk of infection given the pathogen and dose. Purpose of this app is to explore different pathogens and their risks of infection.

Application runs on Streamlit and pulls all data from the QMRA website endpoint. You can customize the application to pull from other sources, as the original version pulled from MongoDB. 

## Installation 

In order to install all the required packages for the application when inside the root directory of the project run: \
```pip install -r requirements.txt ```



##  Running the application 

``` streamlit run main.py ```
### Server hosting

`nohup streamlit run main.py &`

The output of the streamlit server will be in a nohup.out file. In order to restart the server after server reboot you need to add this line to the /etc/rc.local file (Subject to change depending on the hosting machine).

`nohup /path/to/command > /dev/null 2>&1 &` 

### Adding SSL certificate

In order to make this application HTTPS compatible and to enable to ability to embed it as an iframe you need to configure the server with the proper SSL certificate. The configuration should placed inside **.streamlit/config.toml**. Here is an example:

`[server]`

`sslCertFile = '<path-to-file>'`

`sslKeyFile = '<path-to-file>'`