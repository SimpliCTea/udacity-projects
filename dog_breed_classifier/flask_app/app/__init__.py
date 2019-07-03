from flask import Flask
import os

# Initialize the app
#instance_path = str(os.path.dirname(os.path.abspath(__file__))) + '/instance'
app = Flask(__name__, instance_relative_config=True)

# Load the views
from app import views

# Load the config file
app.config.from_object('config')
