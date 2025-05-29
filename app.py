from flask import Flask, request, render_template, redirect, url_for, flash, session, jsonify
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
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Added to suppress warning

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
    receiver_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    message = db.relationship('Message', backref='replies')
    sender = db.relationship('User', foreign_keys=[sender_id], backref='sent_replies')
    receiver = db.relationship('User', foreign_keys=[receiver_id], backref='received_replies')


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


# Prediction Model (moved up with other models)
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    age = db.Column(db.Integer, nullable=False)
    bmi = db.Column(db.Float, nullable=False)
    blood_pressure = db.Column(db.Float, nullable=False)
    cholesterol = db.Column(db.Float, nullable=False)
    result = db.Column(db.Float, nullable=False)

# Notification Model
class Notification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    message = db.Column(db.String(500), nullable=False)
    is_read = db.Column(db.Boolean, default=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    notification_type = db.Column(db.String(50))  # e.g., 'message', 'alert', 'reminder'

    user = db.relationship('User', backref='notifications')


# Flask-Login User Loader
@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


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
    replies = Reply.query.filter(Reply.message_id.in_([msg.id for msg in sent_messages])).all()
    patients = db.session.query(User, HealthData).filter(
        User.role == 'Patient',
        User.id == HealthData.patient_id
    ).all()

    return render_template('doctor_dashboard.html',
                           sent_messages=sent_messages,
                           replies=replies,
                           patients=patients)


@app.route('/patient_dashboard')
@login_required
def patient_dashboard():
    patient_id = current_user.id
    messages = Message.query.filter_by(receiver_id=patient_id).all()
    health_data = HealthData.query.filter_by(patient_id=patient_id).first()
    return render_template('patient_dashboard.html', messages=messages, health_data=health_data)


# Submit Health Data
@app.route('/submit_health_data', methods=['POST'])
@login_required
def submit_health_data():
    if current_user.role != 'Patient':
        flash("Unauthorized access!", "error")
        return redirect(url_for('login'))

    diabetes = request.form.get('diabetes')
    blood_pressure = request.form.get('blood_pressure')
    sugar = request.form.get('sugar')
    hypertension = True if request.form.get('hypertension') == 'on' else False

    health_data = HealthData(
        patient_id=current_user.id,
        diabetes=diabetes,
        blood_pressure=blood_pressure,
        hypertension=hypertension,
        sugar=sugar
    )
    db.session.add(health_data)
    db.session.commit()

    flash("Health data submitted successfully!", "success")
    return redirect(url_for('patient_dashboard'))


@app.route('/send_message', methods=['POST'])
@login_required
def send_message():
    if current_user.role != 'Doctor':
        flash("Unauthorized access!", "error")
        return redirect(url_for('login'))

    patient_id = request.form.get('patient_id')
    message_content = request.form.get('message')

    if not patient_id or not message_content:
        flash("Missing required fields!", "error")
        return redirect(url_for('doctor_dashboard'))

    new_message = Message(
        sender_id=current_user.id,
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
@login_required
def reply_message():
    message_id = request.form.get('message_id')
    reply_content = request.form.get('reply_content')

    if not message_id or not reply_content:
        return jsonify({'success': False, 'error': 'Missing required fields.'}), 400

    original_message = Message.query.get(message_id)
    if not original_message:
        return jsonify({'success': False, 'error': 'Original message not found.'}), 404

    receiver_id = original_message.sender_id

    reply = Reply(
        content=reply_content,
        timestamp=datetime.utcnow(),
        message_id=message_id,
        sender_id=current_user.id,
        receiver_id=receiver_id
    )
    db.session.add(reply)
    db.session.commit()

    return jsonify({'success': True, 'message': 'Reply sent successfully!'})


@app.route('/delete-message', methods=['POST'])
@login_required
def delete_message():
    message_id = request.json.get('id')

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
@login_required
def clear_all_messages():
    if current_user.role != 'Admin':
        return jsonify({"success": False, "error": "Unauthorized"}), 403

    try:
        Message.query.delete()
        db.session.commit()
        return jsonify({"success": True})
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/send_health_alert', methods=['POST'])
@login_required
def send_health_alert():
    if current_user.role != 'Doctor':
        flash("Unauthorized access!", "error")
        return redirect(url_for('login'))

    health_alert_data = request.form.get('health_alert_data')
    print(f"Health alert received: {health_alert_data}")
    return redirect(url_for('doctor_dashboard'))


@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        age = int(request.form.get('age'))
        bmi = float(request.form.get('bmi'))
        blood_pressure = float(request.form.get('blood_pressure'))
        cholesterol = float(request.form.get('cholesterol'))

        input_data = np.array([[age, bmi, blood_pressure, cholesterol]])
        prediction = predict_health_risk(input_data)[0]

        new_prediction = Prediction(
            age=age,
            bmi=bmi,
            blood_pressure=blood_pressure,
            cholesterol=cholesterol,
            result=prediction
        )
        db.session.add(new_prediction)
        db.session.commit()

        return redirect(url_for('doctor_dashboard'))
    except Exception as e:
        flash(f"Error in prediction: {str(e)}", "error")
        return redirect(url_for('patient_dashboard'))


@app.route('/simulate', methods=['POST'])
@login_required
def simulate():
    try:
        age = int(request.form.get('age'))
        initial_infected = int(request.form.get('initial_infected'))
        contact_rate = float(request.form.get('contact_rate'))
        recovery_rate = float(request.form.get('recovery_rate'))
        disease_duration = int(request.form.get('disease_duration'))

        population = 1000
        initial_susceptible = population - initial_infected

        def sir_model(y, t, beta, gamma):
            S, I, R = y
            dSdt = -beta * S * I / population
            dIdt = beta * S * I / population - gamma * I
            dRdt = gamma * I
            return [dSdt, dIdt, dRdt]

        S0 = initial_susceptible
        I0 = initial_infected
        R0 = 0

        t = np.arange(0, disease_duration + 1, 1)

        sol = odeint(sir_model, [S0, I0, R0], t, args=(contact_rate, recovery_rate))

        susceptible = sol[:, 0].tolist()
        infected = sol[:, 1].tolist()
        recovered = sol[:, 2].tolist()

        max_infected = max(infected)
        peak_day = t[np.argmax(infected)]
        end_day = t[-1]
        total_recovered = round(recovered[-1])

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
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/notifications')
@login_required
def notifications():
    user_notifications = Notification.query.filter_by(user_id=current_user.id) \
        .order_by(Notification.timestamp.desc()).all()

    # Mark notifications as read when viewed
    for notification in user_notifications:
        if not notification.is_read:
            notification.is_read = True
    db.session.commit()

    return render_template('notifications.html', notifications=user_notifications)


# Helper function to create notifications
def create_notification(user_id, message, notification_type=None):
    notification = Notification(
        user_id=user_id,
        message=message,
        notification_type=notification_type
    )
    db.session.add(notification)
    db.session.commit()
    return notification


# Statistics Route
@app.route('/statistics')
@login_required
def statistics():
    # Patient statistics
    total_patients = User.query.filter_by(role='Patient').count()

    # Health data statistics
    health_stats = {
        'diabetes': {
            'yes': HealthData.query.filter(HealthData.diabetes == 'Yes').count(),
            'no': HealthData.query.filter(HealthData.diabetes == 'No').count()
        },
        'hypertension': {
            'yes': HealthData.query.filter_by(hypertension=True).count(),
            'no': HealthData.query.filter_by(hypertension=False).count()
        }
    }

    # Message statistics
    message_stats = {
        'total_messages': Message.query.count(),
        'unreplied_messages': Message.query.filter(~Message.replies.any()).count(),
        'replies_sent': Reply.query.count()
    }

    # Calculate reply percentage safely
    if message_stats['total_messages'] > 0:
        reply_percentage = (message_stats['replies_sent'] / message_stats['total_messages']) * 100
    else:
        reply_percentage = 0

    # Risk prediction statistics
    risk_stats = db.session.query(
        db.func.avg(Prediction.result).label('avg_risk'),
        db.func.max(Prediction.result).label('max_risk'),
        db.func.min(Prediction.result).label('min_risk')
    ).first()

    return render_template('statistics.html',
                           total_patients=total_patients,
                           health_stats=health_stats,
                           message_stats=message_stats,
                           reply_percentage=reply_percentage,
                           risk_stats=risk_stats)

    # Delete single notification


@app.route('/delete_notification/<int:notification_id>', methods=['DELETE'])
@login_required
def delete_notification(notification_id):
    notification = Notification.query.get_or_404(notification_id)
    if notification.user_id != current_user.id:
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403

    db.session.delete(notification)
    db.session.commit()
    return jsonify({'success': True})


# Clear all notifications
@app.route('/clear_notifications', methods=['DELETE'])
@login_required
def clear_notifications():
    Notification.query.filter_by(user_id=current_user.id).delete()
    db.session.commit()
    return jsonify({'success': True})



if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)