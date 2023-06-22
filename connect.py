import pymongo 
import streamlit as st
def connect_to_db(**kwargs):
    client = pymongo.MongoClient(**st.secrets["mongo"])
    db = client['RiskApp']
    return db

def get_pathogens(db, **kwargs):
    collection = db['pathogens']
    items = collection.find()
    items = list(items)
    return items 
