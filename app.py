from flask import Flask, request, jsonify, render_template, redirect, url_for, send_from_directory, session
import os
import tensorflow as tf
from werkzeug.utils import secure_filename
import smtplib
from email.message import EmailMessage
import random
import requests
import jwt
import datetime
import numpy as np
import traceback
from pymongo import MongoClient
from pymongo.errors import PyMongoError  # Add this
from flask_cors import CORS
from urllib.parse import quote, unquote
from werkzeug.security import generate_password_hash, check_password_hash
import bcrypt


app = Flask(__name__)

app.secret_key = os.urandom(24) 

CORS(app)
# MongoDB connection
client = MongoClient("mongodb+srv://mahikaroy2004:YZi4loEfL43HimUB@clustor0.66sfs.mongodb.net/?retryWrites=true&w=majority&appName=clustor0")  # Replace with MongoDB URI if using Atlas
db = client["medical_app"]
doctors_collection = db["doctors"]
appointments_collection = db["appointments"]
patients_collection = db["patients"]

# Add an empty list for ratings if it doesn't exist
doctors_collection.update_many({}, {"$set": {"ratings": []}}, upsert=False)



# ZegoCloud API Credentials
APP_ID = 2002785589
SERVER_SECRET = "cebe630985532cf6de2dda0d417610b6"

EMAIL_USER= "devlearning319@gmail.com"
EMAIL_PASSWORD= "bmok jgvg mxlt igwc"  # Not your Gmail password!

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

doctors = []  # Stores registered doctor details
appointments = {}  # Stores booked appointments

#Load ML models (add error handling)
try:
    organ_model = tf.keras.models.load_model('C:/Users/asus/Desktop/ai_py/frosthackAi_py/Ultraclassif_mobilenet.keras')
    breast_model = tf.keras.models.load_model('C:/Users/asus/Desktop/ai_py/frosthackAi_py/BreastModelB7.keras')
    ovary_model = tf.keras.models.load_model('C:/Users/asus/Desktop/ai_py/frosthackAi_py/bestmodel.keras')
except Exception as e:
    print(f"Error loading ML models: {e}")
    exit(1)

# Home page - Choose Patient or Doctor
@app.route('/')
def select_user():
    return render_template('login.html')
# Patient Login
@app.route("/patient_login", methods=["GET"])
def patient_login_page():
    return render_template("patient_login.html")


@app.route('/patient_login', methods=['GET', 'POST'])
def patient_login():
    if request.method == 'POST':
        data = request.json
        email = data.get("email")
        password = data.get("password")
        
        patient = client.db.patients_collection.find_one({"email": email})
        if patient and check_password_hash(patient["password"], password):
            session['patient_id'] = str(patient['_id'])
            return jsonify({"success": True})
        else:
            return jsonify({"success": False, "message": "Invalid email or password"})
    return render_template('patient_login.html')


@app.route('/patient_register', methods=['GET', 'POST'])
def patient_register():
    if request.method == 'POST':
        data = request.json
        name = data.get("name")
        email = data.get("email")
        password = data.get("password")
        
        if client.db.patients_collection.find_one({"email": email}):
            return jsonify({"success": False, "message": "Email already exists"})
        
        hashed_password = generate_password_hash(password)
        client.db.patients_collection.insert_one({"name": name, "email": email, "password": hashed_password})
        return jsonify({"success": True})
    return render_template('patient_register.html')


# Patient Home Page
@app.route('/patient_home')
def patient_home():
    return render_template('patient_home.html')

# Doctor Registration Page
@app.route('/doctor_register', methods=['GET', 'POST'])
def doctor_register():
    if request.method == 'POST':
        try:
            # Get form data
            name = request.form['name']
            email = request.form['email']
            specialization = request.form.get('specialization', 'General Practice')
            available_times = request.form.getlist('available_times')

            # Validate required fields
            if not name or not email:
                return "Name and Email are required fields", 400

            # Create doctor document
            doctor_data = {
                "name": name,
                "email": email,
                "specialization": specialization,
                "available_times": available_times,
                "registration_date": datetime.datetime.utcnow()
            }

            # Upsert doctor data
            result = doctors_collection.update_one(
                {"email": email},
                {"$set": doctor_data},
                upsert=True
            )

            return redirect(url_for('doctor_consultation'))

        except Exception as e:
            return f"An error occurred: {str(e)}", 500

    return render_template('doctor_register.html')

@app.route('/submit_rating', methods=['POST'])
def submit_rating():
    data = request.json
    doctor_email = data.get('doctor_email')
    rating = int(data.get('rating'))

    if not doctor_email or not (1 <= rating <= 5):
        return jsonify({"success": False, "message": "Invalid rating input."})

    doctor = doctors_collection.find_one({"email": doctor_email})
    if not doctor:
        return jsonify({"success": False, "message": "Doctor not found."})

    # Update doctor rating list
    doctors_collection.update_one(
        {"email": doctor_email},
        {"$push": {"ratings": rating}}
    )

    # Recalculate average rating
    updated_doctor = doctors_collection.find_one({"email": doctor_email})
    avg_rating = round(sum(updated_doctor.get("ratings", [])) / len(updated_doctor.get("ratings", [])), 1)

    return jsonify({"success": True, "new_avg": avg_rating})


# Display Doctor List for Patients
@app.route('/doctor_consultation')
def doctor_consultation():
    doctors = list(doctors_collection.find({}, {'_id': 0}))
    for doctor in doctors:
        ratings = doctor.get("ratings", [])
        doctor["average_rating"] = round(sum(ratings) / len(ratings), 1) if ratings else 0  # Exclude MongoDB's _id field
    return render_template('doctor_con.html', doctors=doctors)

def send_email(to_email, subject, message):
    try:
        msg = EmailMessage()
        msg.set_content(message)
        msg['Subject'] = subject
        msg['From'] = f"Healthcare System <{EMAIL_USER}>"
        msg['To'] = to_email

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_USER, EMAIL_PASSWORD)
            server.send_message(msg)
            print(f"Email sent to {to_email}")
    except Exception as e:
        print(f"Email failed to {to_email}: {str(e)}\n{traceback.format_exc()}")

# Patient selects an appointment

@app.route('/book_appointment', methods=['POST'])
def book_appointment():
    retries = 3
    for attempt in range(retries):
        try:
            data = request.get_json()
            doctor_name = data['doctor_name']
            selected_time = data['selected_time']
            patient_email = data['patient_email']

            meeting_id = f"{doctor_name.replace(' ', '_')}_{patient_email.replace('@', '_')}"
            safe_doctor = quote(doctor_name.strip().replace(' ', '_'), safe='')
            safe_patient = quote(patient_email.strip().replace('@', '_').replace(':', '_'), safe='')
            meeting_id = f"{safe_doctor}_{safe_patient}"
            # Atomic findAndModify operation
            doctor = doctors_collection.find_one_and_update(
                {
                    "name": doctor_name,
                    "available_times": selected_time
                },
                {
                    "$pull": {"available_times": selected_time}
                },
                return_document=True,
                session=None  # Remove session for atomic operation
            )

            if not doctor:
                return jsonify({'error': 'Time slot not available'}), 400

            # Create appointment without transaction
            appointments_collection.insert_one({
            "doctor_name": doctor_name,
            "patient_email": patient_email,
            "time": selected_time,
            "status": "booked",
            "meeting_id": meeting_id,  # Add meeting_id here
            "created_at": datetime.datetime.utcnow()
            })

            meet_link = url_for('video_meet', room_id=meeting_id, _external=True)  # Add room_id parameter
            
            # Send emails
            send_email(
                doctor["email"],
                "Appointment Booked",
                f"Your time slot {selected_time} has been booked. Meeting Link: {meet_link}"
            )
            send_email(
                patient_email,
                "Appointment Confirmed",
                f"Your appointment is confirmed. Meeting Link: {meet_link}"
            )

            return jsonify({
                'message': 'Appointment booked successfully!',
                'meet_link': meet_link
            })

        except PyMongoError as e:
            if attempt < retries - 1 and e.has_error_label('TransientTransactionError'):
                print(f"Retrying transaction (attempt {attempt + 1})")
                continue
            return jsonify({'error': str(e)}), 500
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        
# ML Detection Page
@app.route('/ml_detection')
def ml_detection():
    return render_template('ml_detection.html')

# Upload and Process Image
# Upload and Process Image
@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Load and preprocess image
        img = tf.keras.preprocessing.image.load_img(filepath)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        
        # Initialize prediction variables
        organ = "Unknown"
        classification = "Unknown"
        
        # Organ classification
        organ_img = tf.image.resize(img_array, [224, 224])/255.0
        organ_img = tf.expand_dims(organ_img, axis=0)  # Add batch dimension
        organ_pred = organ_model(organ_img, training=False)
        organ = "Ovary" if organ_pred[0][0] > 0.5 else "Breast"

        # Disease classification
        if organ == "Breast":
            processed_img = tf.image.resize(img_array, [300, 300])/255.0
            processed_img = tf.expand_dims(processed_img, axis=0)
            pred = breast_model(processed_img, training=False)
            classes = ["Benign", "Malignant", "Normal"]
            classification = classes[tf.argmax(pred[0]).numpy()]
        else:
            processed_img = tf.image.resize(img_array, [224, 224])/255.0
            processed_img = tf.expand_dims(processed_img, axis=0)
            pred = ovary_model(processed_img, training=False)
            classification = "PCOS Detected" if pred[0][0] > 0.5 else "No PCOS Detected"


        return jsonify({
            "success": True,
            "organ": organ,
            "classification": classification,
            "image_url": f"/uploads/{filename}"
        })

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
   
@app.route('/uploads/<filename>')
def serve_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/generate-zego-token')
def generate_zego_token():
    try:
        room_id = request.args.get('roomID')
        user_id = request.args.get('userID')
        
        payload = {
            "app_id": APP_ID,
            "room_id": room_id,
            "user_id": user_id,
            "privilege": {  # Add proper privileges
                "1": 1,  # Login permission
                "2": 1   # Publish stream permission
            },
            "exp": int((datetime.datetime.utcnow() + datetime.timedelta(hours=2)).timestamp())
        }
        
        token = jwt.encode(payload, SERVER_SECRET, algorithm="HS256")
        return jsonify({"token": token})
        
    except Exception as e:
        print(f"Token generation error: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
# Generate Video Meet Link
@app.route('/start_meeting/<doctor_name>/<patient_email>')
def start_meeting(doctor_name, patient_email):
    try:
        # Verify appointment exists
        appointment = appointments_collection.find_one({
            "doctor_name": doctor_name,
            "patient_email": patient_email,
            "status": "booked"
        })
        
        if not appointment:
            return jsonify({"error": "No valid appointment found"}), 404
            
        # Generate meeting link
        meet_link = generate_meet_link(doctor_name, patient_email)

        # Notify doctor via email
        doctor = doctors_collection.find_one({"name": doctor_name})
        send_email(doctor["email"], f"Your video consultation meeting link: {meet_link}")

        return redirect(meet_link)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    
    # return jsonify({'error': 'No appointment found!'})

# Update video_meet route
@app.route('/video_meet/<room_id>')
def video_meet(room_id):
    try:
        #decoded_room_id = unquote(room_id)
        # Verify appointment exists
        appointment = appointments_collection.find_one({
            "meeting_id": room_id,
            "status": "booked"
        })
        
        if not appointment:
            return render_template("error.html", message="Invalid meeting ID")

        return render_template("web_uikit_1.html", room_id=room_id)
    except Exception as e:
        print(f"Video meet error: {str(e)}\n{traceback.format_exc()}")
        return render_template("error.html", message="Failed to initialize meeting")
    

def generate_meet_link(doctor_name, patient_email):
    api_url = "https://api.zegocloud.com/v1/rooms"

    # Generate a unique room_id from doctor and patient details
    room_id = f"{doctor_name.replace(' ', '')}_{patient_email.replace('@', '')}"

    # Use doctor and patient emails for unique user IDs (instead of just doctor_name)
    doctor_user_id = f"doctor{doctor_name.replace(' ', '')}"
    patient_userid = f"patient{patient_email.replace('@', '')}"

    # Create a Zego meeting room for both doctor and patient
    payload = {
        "app_id": APP_ID,
        "room_id": room_id,
        "user_id": doctor_user_id,  # Doctor's user ID
        "role": "host"
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {SERVER_SECRET}"
    }

    response = requests.post(api_url, json=payload, headers=headers)

    if response.status_code == 200:
        # Return the meeting URL
        return response.json().get("room_url")
    else:
        return "Error generating meeting link"


from app import app
print(app.url_map)


if __name__ == '__main__':
    app.run(debug=True)