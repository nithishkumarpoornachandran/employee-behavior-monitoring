# Standard Library Imports 
import os
import cv2
from datetime import datetime, timedelta
import json
import threading
import numpy as np
import io
import base64
import time
import uuid
import smtplib
import schedule
from collections import defaultdict
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from threading import Thread

# Third-party Libraries
from ultralytics import YOLO                    
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash
from fpdf import FPDF                         
from matplotlib import pyplot as plt           
from twilio.rest import Client                  

# Matplotlib Configuration for Headless Servers
import matplotlib
matplotlib.use('Agg')  # Prevents GUI errors when generating plots in headless environments

# Twilio configuration
TWILIO_ACCOUNT_SID = "AC11e25c009fdf6e90a9ee09c6958e2901"
TWILIO_AUTH_TOKEN = "e0fd84d8ab79b461007bb7f83aa9db98"
TWILIO_PHONE_NUMBER = "whatsapp:+14155238886"
ALERT_PHONE_NUMBER = "whatsapp:+919342353217"

# Email configuration
EMAIL_HOST = 'smtp.gmail.com'
EMAIL_PORT = 587
EMAIL_USER = 'nithishkumarpoornachandran@gmail.com'
EMAIL_PASSWORD = 'obpj cfsf xivz juib'
REPORT_RECIPIENT = '953621106071@ritrjpm.ac.in'

twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Flask App Initialization

app = Flask(__name__)
app.secret_key = "your_secret_key"  

# User credentials
USERS = {
    "admin": generate_password_hash("123")
}

# Load YOLO model
model = YOLO("A:\NK\git\NEW\model\best.pt")
names = model.names

# Directory for activity logs
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Dictionary to track when objects were last logged
last_logged_time = {}

# Time threshold (in seconds) for logging an activity
LOG_THRESHOLD = 10

# Lock for managing concurrent file writes
log_lock = threading.Lock()

# Initialize the in-memory activity log
activity_log = []

# Function to get the log file path
def get_log_file_path(date=None):
    """Get the log file path for a specific date or today."""
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(LOG_DIR, f"{date}.json")

# Setup for the app
@app.before_request
def setup():
    """Check if the model is loaded properly before each request."""
    if not model:
        return jsonify({"status": "error", "message": "YOLO model is not loaded."}), 500

# Authentication routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    """Render login page and handle login."""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username in USERS and check_password_hash(USERS[username], password):
            session['user'] = username
            return redirect(url_for('index'))
        return render_template('login.html', error="Invalid username or password")

    return render_template('login.html')

@app.route('/logout')
def logout():
    """Log the user out."""
    session.pop('user', None)
    return redirect(url_for('login'))

def login_required(func):
    """Decorator to protect routes that require login."""
    def wrapper(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    return wrapper

# Main Page with log selection
@app.route('/')
@login_required
def index():
    """Render the main page with a dropdown for selecting past logs."""
    log_files = [f.replace(".json", "") for f in os.listdir(LOG_DIR) if f.endswith(".json")]
    return render_template('index.html', log_dates=sorted(log_files))

activity_counts = {}
ALERT_THRESHOLD = 1
ALERT_COOLDOWN = 60  # seconds
last_alert_time = {}

def send_sms_alert(activity):
    if activity.lower() == "working":
        return

    if activity not in activity_counts:
        activity_counts[activity] = 0
    activity_counts[activity] += 1

    current_time = time.time()

    if activity_counts[activity] >= ALERT_THRESHOLD:
        if activity in last_alert_time:
            if current_time - last_alert_time[activity] < ALERT_COOLDOWN:
                return

        message_body = (
            f"Alert: '{activity}' detected at "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\n\n"
            f"View live feed:\n"
            f"http://192.168.219.47:8080"
        )

        twilio_client.messages.create(
            body=message_body,
            from_=TWILIO_PHONE_NUMBER,
            to=ALERT_PHONE_NUMBER
        )
        print(f"SMS alert sent for {activity}!")

        activity_counts[activity] = 0
        last_alert_time[activity] = current_time

# Function to capture raw video feed
def generate_raw_feed():
    """Generate raw video feed without detection."""
    cap = cv2.VideoCapture("rtsp://admin:XCDOUN@192.168.137.89:554/h264_stream")
    if not cap.isOpened():
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 500), interpolation=cv2.INTER_LINEAR)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

latest_frame = None
lock = threading.Lock()

# Object detection from video file
def detect_objects_from_video():
    cap = cv2.VideoCapture("A:\NK\git\NEW\input\Input.mp4")
    last_logged_time = {}
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 3 != 0:
            continue

        frame = cv2.resize(frame, (640, 500), interpolation=cv2.INTER_LINEAR)
        detected_activities = []

        results = model.track(frame, persist=True)
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, class_id, track_id in zip(boxes, class_ids, track_ids):
                c = names[class_id]
                x1, y1, x2, y2 = box

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{track_id} - {c}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

                current_time = datetime.now()
                if track_id not in last_logged_time or \
                        (current_time - last_logged_time[track_id]).total_seconds() >= LOG_THRESHOLD:
                    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
                    activity = {"timestamp": timestamp, "class": c, "track_id": track_id}
                    detected_activities.append(activity)
                    last_logged_time[track_id] = current_time
                    send_sms_alert(c)

        if detected_activities:
            save_activity_log(detected_activities)

        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Start detection thread
detection_thread = threading.Thread(target=detect_objects_from_video, daemon=True)
detection_thread.start()

@app.route('/video_feed')
@login_required
def video_feed():
    """Video feed route."""
    return Response(generate_raw_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def save_activity_log(detected_activities):
    """Save detected activities to the current day's log file."""
    log_file = get_log_file_path()

    try:
        with log_lock:  
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    log_data = json.load(f)
            else:
                log_data = []

            log_data.extend(detected_activities)

            with open(log_file, "w") as f:
                json.dump(log_data, f, indent=4)
    except Exception as e:
        print(f"Error saving activity log: {e}")

@app.route('/get_logs_by_date', methods=['POST'])
@login_required
def get_logs_by_date():
    """Return activity logs for a specific date."""
    date = request.form.get("date")
    if not date:
        return jsonify({"status": "error", "message": "Date is required."}), 400

    log_file = get_log_file_path(date)
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            return jsonify(json.load(f))
    return jsonify([])

@app.route('/clear_activity_log', methods=['POST'])
@login_required
def clear_activity_log():
    """Clear today's activity log."""
    log_file = get_log_file_path()
    if os.path.exists(log_file):
        os.remove(log_file)
    global activity_log
    activity_log = []
    return jsonify({"status": "success", "message": "Activity log cleared."})

@app.route('/activity_graph', methods=['GET', 'POST'])
@login_required
def activity_graph():
    """Render graphs of the activities for a selected date and handle report generation."""
    activity_data = None
    error = None
    success = None
    selected_date = None

    if request.method == 'POST':
        if 'weekly_report' in request.form:
            try:
                # Generate and send the 7-day report via email
                send_7_days_report_via_email()
                success = "7-day report has been sent via email successfully!"
            except Exception as e:
                error = f"Failed to send email: {str(e)}"
            
        elif 'date' in request.form:
            selected_date = request.form['date']
            log_file = get_log_file_path(selected_date)

            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    log_data = json.load(f)

                activity_counts = {}
                for entry in log_data:
                    activity = entry['class']
                    activity_counts[activity] = activity_counts.get(activity, 0) + 1

                plt.figure(figsize=(6, 4))
                plt.bar(activity_counts.keys(), activity_counts.values(), color='skyblue')
                plt.xlabel("Activity")
                plt.ylabel("Count")
                plt.title("Activity Graph")
                plt.xticks(rotation=45)
                plt.tight_layout()

                img = io.BytesIO()
                plt.savefig(img, format='png')
                img.seek(0)
                encoded_img = base64.b64encode(img.getvalue()).decode()
                activity_data = {"Activity Graph": encoded_img}
                plt.close()
            else:
                error = "No log data available for the selected date."

    return render_template('activity_graph.html', 
                         selected_date=selected_date, 
                         activity_data=activity_data, 
                         error=error,
                         success=success)

@app.route('/download_pdf', methods=['POST'])
@login_required
def download_pdf():
    date = request.form.get('date')
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")
    
    log_file = get_log_file_path(date)
    if not os.path.exists(log_file):
        return "No log data available for the selected date.", 404

    with open(log_file, 'r') as f:
        log_data = json.load(f)

    activity_counts = {}
    for entry in log_data:
        activity = entry['class']
        if activity not in activity_counts:
            activity_counts[activity] = 0
        activity_counts[activity] += 1

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    plt.figure(figsize=(6, 4))
    plt.bar(activity_counts.keys(), activity_counts.values(), color='skyblue')
    plt.xlabel("Activity")
    plt.ylabel("Count")
    plt.title("Activity Graph")
    plt.xticks(rotation=45)
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    import tempfile
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    temp_file.write(img.getvalue())
    temp_file.close()

    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt="Activity Graph", ln=True, align="C")
    pdf.image(temp_file.name, x=10, y=30, w=pdf.w - 20)
    os.unlink(temp_file.name)

    pdf_output = pdf.output(dest='S').encode('latin1')
    return Response(pdf_output, mimetype='application/pdf',
                    headers={"Content-Disposition": f"attachment;filename=ActivityGraph_{date}.pdf"})

# Calculate time spent per activity by comparing timestamps
def calculate_time_spent(activities):
    """Calculate time spent on each activity."""
    time_spent = defaultdict(float)
    for i in range(1, len(activities)):
        prev_time = datetime.strptime(activities[i-1]['timestamp'], "%Y-%m-%d %H:%M:%S")
        curr_time = datetime.strptime(activities[i]['timestamp'], "%Y-%m-%d %H:%M:%S")
        duration = (curr_time - prev_time).total_seconds() / 3600  # in hours
        time_spent[activities[i-1]['class']] += duration
    return time_spent

# Generate report data by reading log files over a date range

def generate_report_data(start_date, end_date):
    """Generate activity summary between two dates."""
    report_data = {
        'working_hours': 0,
        'mobile_usage': 0,
        'other_activities': defaultdict(float),
        'start_date': start_date.strftime("%Y-%m-%d"),
        'end_date': end_date.strftime("%Y-%m-%d")
    }

    current_date = start_date
    while current_date <= end_date:
        log_file = get_log_file_path(current_date.strftime("%Y-%m-%d"))
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                activities = json.load(f)
            
            time_spent = calculate_time_spent(activities)
            for activity, duration in time_spent.items():
                if activity.lower() == 'working':
                    report_data['working_hours'] += duration
                elif 'mobile' in activity.lower():
                    report_data['mobile_usage'] += duration
                else:
                    report_data['other_activities'][activity] += duration
        
        current_date += timedelta(days=1)
    
    return report_data

# Convert report data into a structured PDF

def create_pdf_report(report_data):
    """Create a PDF report from the report data."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Activity Report", 0, 1, 'C')
    pdf.ln(5)
    
    # Date range
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Period: {report_data['start_date']} to {report_data['end_date']}", 0, 1)
    pdf.ln(10)
    
    # Summary
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Summary", 0, 1)
    pdf.set_font("Arial", '', 12)
    
    pdf.cell(0, 10, f"Total Working Hours: {report_data['working_hours']:.2f}", 0, 1)
    pdf.cell(0, 10, f"Total Mobile Usage: {report_data['mobile_usage']:.2f}", 0, 1)
    pdf.ln(5)
    
    # Other activities
    if report_data['other_activities']:
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Other Activities", 0, 1)
        pdf.set_font("Arial", '', 12)
        
        for activity, duration in report_data['other_activities'].items():
            pdf.cell(0, 10, f"{activity}: {duration:.2f} hours", 0, 1)
    
    return pdf

# Attach the report PDF and send via email

def send_email_with_report(pdf, report_data):
    """Send the PDF report via email."""
    msg = MIMEMultipart()
    msg['From'] = EMAIL_USER
    msg['To'] = REPORT_RECIPIENT
    msg['Subject'] = f"Activity Report {report_data['start_date']} to {report_data['end_date']}"
    
    body = f"""Activity Report for {report_data['start_date']} to {report_data['end_date']}
    
Total Working Hours: {report_data['working_hours']:.2f}
Total Mobile Usage: {report_data['mobile_usage']:.2f}
"""
    msg.attach(MIMEText(body, 'plain'))
    
    pdf_attachment = MIMEApplication(pdf.output(dest='S').encode('latin1'), _subtype="pdf")
    pdf_attachment.add_header('Content-Disposition', 'attachment', 
                            filename=f"ActivityReport_{report_data['start_date']}_to_{report_data['end_date']}.pdf")
    msg.attach(pdf_attachment)
    
    try:
        server = smtplib.SMTP(EMAIL_HOST, EMAIL_PORT)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_USER, REPORT_RECIPIENT, msg.as_string())
        server.quit()
        print("Report email sent successfully")
    except Exception as e:
        print(f"Error sending email: {e}")

# Flask Route for Report

@app.route('/generate_report', methods=['POST'])
@login_required
def generate_report():
    """Handle report generation request."""
    report_type = request.form.get('report_type')
    email_report = request.form.get('email_report') == 'true'
    
    end_date = datetime.now()
    if report_type == 'weekly':
        start_date = end_date - timedelta(days=7)
    else:  # monthly
        start_date = end_date - timedelta(days=30)
    
    report_data = generate_report_data(start_date, end_date)
    pdf = create_pdf_report(report_data)
    
    if email_report:
        send_email_with_report(pdf, report_data)
        return jsonify({"status": "success", "message": "Report generated and emailed successfully"})
    else:
        pdf_output = pdf.output(dest='S').encode('latin1')
        filename = f"ActivityReport_{report_type}_{end_date.strftime('%Y%m%d')}.pdf"
        return Response(pdf_output, mimetype='application/pdf',
                       headers={"Content-Disposition": f"attachment;filename={filename}"})

# Sends 7-day work summary with pie chart via email

def send_7_days_report_via_email():
    """Generate 7-day report focusing only on working hours distribution."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    # Fixed working hours assumption (8h/day × 7 days)
    total_working_hours = 56
    
    # Get activities from logs during working hours only
    report_data = generate_report_data(start_date, end_date)
    mobile_hours = report_data['mobile_usage']
    other_hours = sum(report_data['other_activities'].values())
    
    # Calculate working time distribution
    productive_work_hours = total_working_hours - mobile_hours - other_hours
    if productive_work_hours < 0:
        productive_work_hours = 0  # Prevent negative values
    
    # Calculate percentages of working time
    work_percent = (productive_work_hours / total_working_hours) * 100
    mobile_percent = (mobile_hours / total_working_hours) * 100
    other_percent = (other_hours / total_working_hours) * 100
    
    # Create pie chart focused on working hours
    plt.figure(figsize=(8, 6))
    labels = ['Productive Work', 'Mobile Usage', 'Other Activities']
    sizes = [work_percent, mobile_percent, other_percent]
    colors = ['#4CAF50', '#FFC107', '#F44336']
    explode = (0.1, 0, 0)  # emphasize productive work
    
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct=lambda p: f'{p:.1f}%\n({p*56/100:.1f}h)',
            shadow=True, startangle=90)
    plt.title('Working Hours Distribution (56 hours total)', pad=20)
    
    # Save chart
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
    img_buffer.seek(0)
    plt.close()
    
    # Create PDF
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, f"7-Day Work Hours Report ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})", 0, 1, 'C')
    pdf.ln(10)
    
    # Time Distribution Summary
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Working Hours Breakdown (56 hours total)', 0, 1)
    pdf.set_font('Arial', '', 12)
    
    pdf.cell(0, 10, f"Productive Work: {productive_work_hours:.1f}h ({work_percent:.1f}%)", 0, 1)
    pdf.cell(0, 10, f"Mobile Usage: {mobile_hours:.1f}h ({mobile_percent:.1f}%)", 0, 1)
    pdf.cell(0, 10, f"Other Activities: {other_hours:.1f}h ({other_percent:.1f}%)", 0, 1)
    pdf.ln(15)
    
    # Add pie chart
    temp_img = "work_pie.png"
    with open(temp_img, 'wb') as f:
        f.write(img_buffer.getvalue())
    pdf.image(temp_img, x=30, w=150)
    os.remove(temp_img)
    
    # Email content
    msg = MIMEMultipart()
    msg['From'] = EMAIL_USER
    msg['To'] = REPORT_RECIPIENT
    msg['Subject'] = f"7-Day Work Hours Report ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})"
    
    email_body = f"""
    <html>
      <body>
        <h2>7-Day Work Hours Report ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})</h2>
        <h3>Working Hours Distribution (56 hours total)</h3>
        <ul>
          <li><strong>Productive Work:</strong> {productive_work_hours:.1f}h ({work_percent:.1f}%)</li>
          <li><strong>Mobile Usage:</strong> {mobile_hours:.1f}h ({mobile_percent:.1f}%)</li>
          <li><strong>Other Activities:</strong> {other_hours:.1f}h ({other_percent:.1f}%)</li>
        </ul>
        
        <p>See attached PDF for visual breakdown of your working hours.</p>
      </body>
    </html>
    """
    
    msg.attach(MIMEText(email_body, 'html'))
    
    # Attach PDF
    pdf_attachment = MIMEApplication(pdf.output(dest='S').encode('latin1'), _subtype='pdf')
    pdf_attachment.add_header('Content-Disposition', 'attachment',
                            filename=f"WorkHoursReport_{start_date.strftime('%Y%m%d')}.pdf")
    msg.attach(pdf_attachment)
    
    # Send email
    try:
        with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASSWORD)
            server.send_message(msg)
        print("Work hours report sent successfully")
        return True
    except Exception as e:
        print(f"Email sending failed: {str(e)}")
        raise e
    
# Schedule automatic report generation at fixed times

def schedule_reports():
    """Schedule automatic report generation."""
    # Weekly report every Monday at 9 AM
    schedule.every().monday.at("09:00").do(
        lambda: generate_report_data(datetime.now() - timedelta(days=7), datetime.now())
    )
    
    # Monthly report on the 1st at 9 AM
    schedule.every().day.at("09:00").do(
        lambda: generate_report_data(datetime.now() - timedelta(days=30), datetime.now()) 
        if datetime.now().day == 1 else None
    )
    
    while True:
        schedule.run_pending()
        time.sleep(60)

# Start report scheduling thread
report_thread = Thread(target=schedule_reports, daemon=True)
report_thread.start()

if __name__ == '__main__':
    app.run('0.0.0.0', debug=False, port=8080)
