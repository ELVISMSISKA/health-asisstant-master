from flask import Flask, request, render_template, redirect, url_for, flash, session,jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from health_model import predict_health_risk
from scipy.integrate import odeint



app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SECRET_KEY'] = '579276e336632992782762'

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'


# User Model
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), nullable=False)


# Message Model
class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String(500), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    sender_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    receiver_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    sender = db.relationship('User', foreign_keys=[sender_id], backref='sent_messages')
    receiver = db.relationship('User', foreign_keys=[receiver_id], backref='received_messages')


class Reply(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String(500), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    message_id = db.Column(db.Integer, db.ForeignKey('message.id'), nullable=False)
    sender_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    receiver_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)  # New field for reply recipient

    message = db.relationship('Message', backref='replies')
    sender = db.relationship('User', foreign_keys=[sender_id], backref='sent_replies')
    receiver = db.relationship('User', foreign_keys=[receiver_id], backref='received_replies')  # Relationship for receiver


# Health Data Model
class HealthData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    diabetes = db.Column(db.String(100), nullable=True)
    blood_pressure = db.Column(db.String(100), nullable=True)
    hypertension = db.Column(db.Boolean, nullable=True)
    sugar = db.Column(db.String(100), nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    patient = db.relationship('User', backref='health_data')
    


# Flask-Login User Loader
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


@app.route('/')
def home_root():
    return render_template('home.html')


# Signup Route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        role = request.form['role']

        if User.query.filter_by(email=email).first() or User.query.filter_by(username=username).first():
            flash("Username or email already exists!", "error")
            return redirect(url_for('signup'))

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, email=email, password=hashed_password, role=role)
        db.session.add(new_user)
        db.session.commit()
        flash("Signup successful! Please log in.", "success")
        return redirect(url_for('login'))

    return render_template('signup.html')


# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            login_user(user)
            flash(f"Welcome {user.username}!", "success")
            if user.role == "Admin":
                return redirect(url_for('admin_dashboard'))
            elif user.role == "Doctor":
                return redirect(url_for('doctor_dashboard'))
            return redirect(url_for('patient_dashboard'))

        flash("Invalid email or password. Please try again.", "error")
        return redirect(url_for('login'))

    return render_template('login.html')


# Logout Route
@app.route('/logout')
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for('home_root'))


# Admin Dashboard
@app.route('/admin_dashboard')
@login_required
def admin_dashboard():
    if current_user.role != "Admin":
        flash("Unauthorized access!", "error")
        return redirect(url_for('login'))
    users = User.query.all()
    return render_template('admin_dashboard.html', users=users)


# Doctor Dashboard
@app.route('/doctor_dashboard')
@login_required
def doctor_dashboard():
    if current_user.role != 'Doctor':
        flash("Unauthorized access!", "error")
        return redirect(url_for('login'))

    sent_messages = Message.query.filter_by(sender_id=current_user.id).all()
    
    # Fetch replies for messages sent by the doctor
    replies = Reply.query.filter(Reply.message_id.in_([msg.id for msg in sent_messages])).all()
    
    # Fetch patients with their health data
    patients = db.session.query(User, HealthData).filter(
        User.role == 'Patient',
        User.id == HealthData.patient_id
    ).all()

    # Pass replies to the template
    return render_template('doctor_dashboard.html',
                           sent_messages=sent_messages,
                           replies=replies,
                           patients=patients)




@app.route('/patient_dashboard')
@login_required
def patient_dashboard():
    # Get the patient id (assuming you are using flask-login)
    patient_id = current_user.id
    # Fetch messages for the patient
    messages = Message.query.filter_by(receiver_id=patient_id).all()
    # Fetch the health data (make sure this data exists in your database and model)
    health_data = HealthData.query.filter_by(patient_id=patient_id).first()
    # Return the rendered template with both messages and health_data
    return render_template('patient_dashboard.html', messages=messages, health_data=health_data)

# Submit Health Data
@app.route('/submit_health_data', methods=['POST'])
@login_required
def submit_health_data():
    if current_user.role != 'Patient':
        flash("Unauthorized access!", "error")
        return redirect(url_for('login'))

    diabetes = request.form['diabetes']
    blood_pressure = request.form['blood_pressure']
    sugar = request.form['sugar']

    health_data = HealthData(patient_id=current_user.id, diabetes=diabetes, blood_pressure=blood_pressure, sugar=sugar)
    db.session.add(health_data)
    db.session.commit()

    flash("Health data submitted successfully!", "success")
    return redirect(url_for('patient_dashboard'))


@app.route('/send_message', methods=['POST'])
@login_required
def send_message():
    if request.method == 'POST':
        # Get the patient id and message content from the form
        patient_id = request.form.get('patient_id')
        message_content = request.form.get('message')

        # Get the doctor (sender) from the session or current user
        doctor_id = current_user.id

        # Create a new message record in the database
        new_message = Message(
            sender_id=doctor_id,
            receiver_id=patient_id,
            content=message_content,
            timestamp=datetime.utcnow()
        )
        
        db.session.add(new_message)
        db.session.commit()

        flash('Message sent successfully!', 'success')

        return redirect(url_for('doctor_dashboard'))

# Route to handle replies
@app.route('/reply_message', methods=['POST'])
def reply_message():
    message_id = request.form['message_id']
    reply_content = request.form['reply_content']

    # Fetch the original message to determine the reply recipient
    original_message = Message.query.get(message_id)
    if not original_message:
        return jsonify({'success': False, 'error': 'Original message not found.'})

    # Set the receiver as the original sender of the message (doctor)
    receiver_id = original_message.sender_id

    # Create and save the reply
    reply = Reply(
        content=reply_content,
        timestamp=datetime.utcnow(),
        message_id=message_id,
        sender_id=current_user.id,  # Current user (patient) is the sender
        receiver_id=receiver_id    # Original sender (doctor) is the receiver
    )
    db.session.add(reply)
    db.session.commit()

    return jsonify({'success': True, 'message': 'Reply sent successfully!'})



@app.route('/delete-message', methods=['POST'])
def delete_message():
    message_id = request.json.get('id')  # Use .json for JSON data

    # Check if message_id was provided
    if not message_id:
        return jsonify({"success": False, "error": "Message ID not provided"}), 400

    message = Message.query.get(message_id)
    
    if message:
        try:
            db.session.delete(message)
            db.session.commit()
            return jsonify({"success": True})
        except Exception as e:
            db.session.rollback()
            return jsonify({"success": False, "error": str(e)}), 500
    return jsonify({"success": False, "error": "Message not found"}), 404

@app.route('/clear-all-messages', methods=['POST'])
def clear_all_messages():
    try:
        Message.query.delete()
        db.session.commit()
        return jsonify({"success": True})
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)})

@app.route('/send_health_alert', methods=['POST'])
def send_health_alert():
    # Check if the request method is POST
    if request.method == 'POST':
        # Here you can extract data from the form (if needed)
        health_alert_data = request.form.get('health_alert_data')
        
        # Your custom logic to handle the health alert data
        # For example, you can log the data or send a notification
        
        print(f"Health alert received: {health_alert_data}")
        
        # After handling the health alert, redirect to the doctor dashboard
        return redirect(url_for('doctor_dashboard'))
    
#####



# Database model
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    age = db.Column(db.Integer, nullable=False)
    bmi = db.Column(db.Float, nullable=False)
    blood_pressure = db.Column(db.Float, nullable=False)
    cholesterol = db.Column(db.Float, nullable=False)
    result = db.Column(db.Float, nullable=False)

# Initialize the database
with app.app_context():
    db.create_all()

@app.route('/')
def index():
    return render_template('patient_dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['age'])
    bmi = float(request.form['bmi'])
    blood_pressure = float(request.form['blood_pressure'])
    cholesterol = float(request.form['cholesterol'])

    # Predict the health risk
    input_data = np.array([[age, bmi, blood_pressure, cholesterol]])
    prediction = predict_health_risk(input_data)[0]

    # Save the prediction to the database
    new_prediction = Prediction(age=age, bmi=bmi, blood_pressure=blood_pressure, cholesterol=cholesterol, result=prediction)
    db.session.add(new_prediction)
    db.session.commit()

    # Redirect to doctor's page
    return redirect(url_for('doctor_dashboard'))


@app.route('/simulate', methods=['POST'])
def simulate():
    # Get data from form
    age = int(request.form['age'])
    initial_infected = int(request.form['initial_infected'])
    contact_rate = float(request.form['contact_rate'])
    recovery_rate = float(request.form['recovery_rate'])
    disease_duration = int(request.form['disease_duration'])

    # Total population size
    population = 1000
    initial_susceptible = population - initial_infected

    # Define SIR model equations
    def sir_model(y, t, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / population
        dIdt = beta * S * I / population - gamma * I
        dRdt = gamma * I
        return [dSdt, dIdt, dRdt]

    # Initial conditions: S0, I0, R0
    S0 = initial_susceptible
    I0 = initial_infected
    R0 = 0  # Initially no one is recovered

    # Time points (days)
    t = np.arange(0, disease_duration + 1, 1)  # Ensure whole numbers (integers)

    # Solve ODE
    sol = odeint(sir_model, [S0, I0, R0], t, args=(contact_rate, recovery_rate))

    # Prepare results
    susceptible = sol[:, 0].tolist()
    infected = sol[:, 1].tolist()
    recovered = sol[:, 2].tolist()

    # Generate description based on simulation results
    max_infected = max(infected)
    peak_day = t[np.argmax(infected)]  # Find the day with the peak infection
    end_day = t[-1]  # Last day of the simulation
    total_recovered = round(recovered[-1])  # Total recovered at the end of the simulation

    description = f"The disease peaked on day {peak_day} with {round(max_infected)} infected individuals. " \
                  f"After day {peak_day}, the number of infected people started to decrease as more individuals recovered. " \
                  f"By the end of the simulation (day {end_day}), {total_recovered} people had recovered. " \
                  f"The disease is expected to subside over time as the number of susceptible individuals decreases."

    return jsonify({
        "time": t.tolist(),
        "susceptible": susceptible,
        "infected": infected,
        "recovered": recovered,
        "description": description
    })


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
