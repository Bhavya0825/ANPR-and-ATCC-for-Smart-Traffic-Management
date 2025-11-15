from flask import Flask, render_template, request, redirect, url_for, jsonify, Response
import os
from werkzeug.utils import secure_filename
import cv2
from app_instance import create_app
from dotenv import load_dotenv
import pymysql

# Load environment variables from .env file
load_dotenv()

# Initialize the app using the create_app function
app = create_app()

# Fetch camera IPs from environment variables (comma-separated list)
camera_ips_env = os.getenv('LIVE_CCTV_IPS', '')
if camera_ips_env:
    app.config['CAMERA_IPS'] = camera_ips_env.split(',')
else:
    app.config['CAMERA_IPS'] = []

UPLOAD_FOLDER = app.config['UPLOAD_FOLDER']
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



def dbconnection():
     connection = pymysql.connect(host='127.0.0.1',database='traffic_management',user='admin',password='12345678')
     return connection


# Routes
@app.route('/')
def login():
    return render_template('login.html')

from dbconnection import get_connection

@app.route('validate_login',methods=['POST'])
def validate_login():
    username = request.form.get('username')
    password =request.form.get('password')
    connection = get_connection()
    print(f"number_plate........{connection}")
    print(username,password)
    with connection.cursor() as cursor:
        #use parameterized query to prevent SQL injection
        sql_query = "SELECT * FROM login_details WHERE username =%s AND password=%s"
        cursor.execute(sql_query,(username,password))
        result = cursor.fetchone()
        if result:
            print(f"User {username} loggied in successfully.")
            return redirect(url_for('home'))
    return render_template('login.html',error="Invalid username or password.")

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/search',METHODS=['GET','POST'])
def search_license_plate():
    number_plate = request.form.get('number_plate') #Get license plate from form
    connection = get_connection()
    print(f"Database Connection:{connection}")
    vehicle_data = None
    error = None
    if request.method == 'POST' and number_plate:
        with connection.cursor(pymysql.cursors.DictCursor) as cursor:
            sql_query = "SELECT * FROM vehicle_data WHERE number_plate = %s"
            cursor.execute(sql_query,(number_plate,))
            result = cursor.fetchone()
            print("SQL Statement Executed:",sql_query)
            if result:
                print(f"Vehicle data found for {number_plate}")
                vehicle_data = result
            else:
                error = "No details found for the entered number  plate."
    return render_template('search.html',vehicle_data=vehicle_data,error=error)
######################################################################

app.config['UPLOAD_FOLDER'] = os.path.json(os.path.dirname(os.path.abspath(__file__)),'static','uploads')
@app.route('/upload_video',methods=['GET','POST'])
def upload_video():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('upload_video.html',error="No file selected.")
        file = request.files['file']
        if file and allowed_file(file.filename):
            #ensure the filename is safe
            filename = secure_filename(file.filename)
            #save the file in the static/uploads directory
            file.save(os.path.join(app.config['UPLOAD_FLODER']),filename)
            return render_template('upload_video.html',success=True,filename=filename)
        else:
            return render_template('upload_video.html',error= "Invalid file type. Allowed : mp4,avi,mov,mkv.")
    return render_template('upload_video.html')


@app.route('/results',method=['GET'])
def result():
    return render_template('results.html')

@app.route('/logout')
def logout()