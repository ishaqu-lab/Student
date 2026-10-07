from pathlib import Path
import io
import math
import os
import uuid
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_file,
    jsonify
)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    logout_user,
    login_required,
    current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from authlib.integrations.flask_client import OAuth
from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer as Serializer

from services import (
    VALID_CATEGORIES,
    VALID_SENTIMENTS,
    load_metadata,
    validate_csv,
    predict_feedback,
    apply_filters
)

# ============================================================
# APPLICATION CONFIGURATION & ENV LOADING
# ============================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)

app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "student-feedback-dev-key")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# PostgreSQL / Database Config (Fixes Render's postgres:// prefix requirement)
db_url = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'users.db'}")
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)
app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Google OAuth Config
app.config["GOOGLE_CLIENT_ID"] = os.getenv("GOOGLE_CLIENT_ID")
app.config["GOOGLE_CLIENT_SECRET"] = os.getenv("GOOGLE_CLIENT_SECRET")

# Mail Config
app.config["MAIL_SERVER"] = os.getenv("MAIL_SERVER", "smtp.gmail.com")
app.config["MAIL_PORT"] = int(os.getenv("MAIL_PORT", 587))
app.config["MAIL_USE_TLS"] = os.getenv("MAIL_USE_TLS", "True") == "True"
app.config["MAIL_USERNAME"] = os.getenv("MAIL_USERNAME")
app.config["MAIL_PASSWORD"] = os.getenv("MAIL_PASSWORD")

# Extensions Initialization
db = SQLAlchemy(app)
mail = Mail(app)
oauth = OAuth(app)

login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message_category = "info"

# Register Google OAuth Client
google = oauth.register(
    name="google",
    client_id=app.config["GOOGLE_CLIENT_ID"],
    client_secret=app.config["GOOGLE_CLIENT_SECRET"],
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"}
)

DATA_PATH = BASE_DIR / "data" / "default_dataset.csv"
UPLOAD_DATA_PATH = BASE_DIR / "uploads" / "current_analysis.csv"

# ============================================================
# DATABASE MODEL
# ============================================================

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=True)
    password = db.Column(db.String(200), nullable=True)

    def get_reset_token(self):
        s = Serializer(app.config["SECRET_KEY"])
        return s.dumps({"user_id": self.id})

    @staticmethod
    def verify_reset_token(token, expires_sec=1800):
        s = Serializer(app.config["SECRET_KEY"])
        try:
            user_id = s.loads(token, max_age=expires_sec)["user_id"]
        except Exception:
            return None
        return User.query.get(user_id)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Auto-create tables in context
with app.app_context():
    db.create_all()

# ============================================================
# LOAD DATA & DASHBOARD HELPERS
# ============================================================

def load_data():
    if UPLOAD_DATA_PATH.exists():
        return pd.read_csv(UPLOAD_DATA_PATH)

    if not DATA_PATH.exists():
        return pd.DataFrame(
            columns=["ID", "Feedback Text", "Category", "Sentiment", "Date"]
        )

    return pd.read_csv(DATA_PATH)

def get_dashboard_data(df=None):
    if df is None:
        df = load_data()

    total = len(df)
    sentiment_counts = (
        df["Sentiment"].value_counts().to_dict() if total else {}
    )

    percentages = {
        sentiment: round(
            (sentiment_counts.get(sentiment, 0) / total) * 100, 1
        )
        if total
        else 0
        for sentiment in VALID_SENTIMENTS
    }

    category_counts = (
        df.groupby(["Category", "Sentiment"])
        .size()
        .unstack(fill_value=0)
        if total
        else pd.DataFrame()
    )

    category_data = []
    for category in VALID_CATEGORIES:
        if not category_counts.empty and category in category_counts.index:
            row = category_counts.loc[category]
            pos = int(row.get("Positive", 0)) if "Positive" in row else 0
            neu = int(row.get("Neutral", 0)) if "Neutral" in row else 0
            neg = int(row.get("Negative", 0)) if "Negative" in row else 0
        else:
            pos, neu, neg = 0, 0, 0

        category_data.append({
            "category": category,
            "Positive": pos,
            "Neutral": neu,
            "Negative": neg,
        })

    return {
        "total": total,
        "sentiment_counts": sentiment_counts,
        "percentages": percentages,
        "category_data": category_data
    }

def send_reset_email(user):
    token = user.get_reset_token()
    msg = Message(
        "Password Reset Request",
        sender=app.config["MAIL_USERNAME"],
        recipients=[user.email]
    )
    msg.body = f"""To reset your password, visit the following link:
{url_for('reset_token', token=token, _external=True)}

If you did not make this request, simply ignore this email.
"""
    mail.send(msg)

# ============================================================
# AUTHENTICATION ROUTES
# ============================================================

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        user = User.query.filter_by(username=username).first()
        if user and user.password and check_password_hash(user.password, password):
            login_user(user)
            flash("Logged in successfully!", "success")
            next_page = request.args.get("next")
            return redirect(next_page or url_for("dashboard"))
        else:
            flash("Login unsuccessful. Please check username and password.", "danger")

    return render_template("login.html", active_page="login")

@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")

        if User.query.filter_by(username=username).first():
            flash("Username already exists.", "warning")
            return redirect(url_for("register"))

        if User.query.filter_by(email=email).first():
            flash("Email already registered.", "warning")
            return redirect(url_for("register"))

        hashed_pw = generate_password_hash(password, method="scrypt")
        new_user = User(username=username, email=email, password=hashed_pw)
        db.session.add(new_user)
        db.session.commit()

        flash("Account created! You can now log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html", active_page="register")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))

@app.route("/login/google")
def google_login():
    redirect_uri = url_for("google_authorize", _external=True)
    return google.authorize_redirect(redirect_uri)

@app.route("/login/google/authorize")
def google_authorize():
    token = google.authorize_access_token()
    user_info = token.get("userinfo")

    if not user_info:
        flash("Failed to log in with Google.", "danger")
        return redirect(url_for("login"))

    email = user_info.get("email")
    username = user_info.get("name", email.split("@")[0])

    user = User.query.filter_by(email=email).first()
    if not user:
        user = User(username=username, email=email, password=None)
        db.session.add(user)
        db.session.commit()

    login_user(user)
    flash("Logged in with Google successfully!", "success")
    return redirect(url_for("dashboard"))

@app.route("/reset_password", methods=["GET", "POST"])
def reset_request():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email = request.form.get("email")
        user = User.query.filter_by(email=email).first()
        if user:
            send_reset_email(user)
            flash("An email with instructions to reset your password has been sent.", "info")
            return redirect(url_for("login"))
        else:
            flash("There is no account with that email.", "danger")

    return render_template("reset_request.html")

@app.route("/reset_password/<token>", methods=["GET", "POST"])
def reset_token(token):
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    user = User.verify_reset_token(token)
    if user is None:
        flash("That is an invalid or expired token.", "warning")
        return redirect(url_for("reset_request"))

    if request.method == "POST":
        password = request.form.get("password")
        user.password = generate_password_hash(password, method="scrypt")
        db.session.commit()
        flash("Your password has been updated! You can now log in.", "success")
        return redirect(url_for("login"))

    return render_template("reset_token.html")

# ============================================================
# APP CORE ROUTES
# ============================================================

@app.route("/")
def home():
    return render_template("home.html", active_page="home")

@app.route("/dashboard")
@login_required
def dashboard():
    df = load_data()
    metadata = load_metadata()
    data = get_dashboard_data(df)

    recent = (
        df.sort_values("ID", ascending=False)
        .head(10)
        .to_dict("records")
        if len(df)
        else []
    )

    return render_template(
        "dashboard.html",
        data=data,
        recent=recent,
        metadata=metadata,
        categories=VALID_CATEGORIES,
        active_page="dashboard"
    )

@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    if request.method == "POST":
        file = request.files.get("file")

        ok, result = validate_csv(file)
        if not ok:
            flash(result, "danger")
            return redirect(url_for("upload"))

        upload_id = uuid.uuid4().hex[:12]
        safe_name = secure_filename(file.filename)
        upload_path = BASE_DIR / "uploads" / f"{upload_id}_{safe_name}"

        upload_path.parent.mkdir(parents=True, exist_ok=True)
        file.save(upload_path)

        try:
            predictions, probabilities = predict_feedback(result["Feedback Text"])
        except Exception as exc:
            upload_path.unlink(missing_ok=True)
            flash(f"Unable to analyse the uploaded feedback: {exc}", "danger")
            return redirect(url_for("upload"))

        result["Sentiment"] = predictions
        if probabilities is not None:
            result["Prediction Confidence"] = [
                round(float(prob), 4) for prob in probabilities
            ]

        UPLOAD_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(UPLOAD_DATA_PATH, index=False)

        flash(
            f"Dataset uploaded and analysed successfully: {len(result)} records.",
            "success"
        )
        return redirect(url_for("analysis"))

    return render_template(
        "upload.html",
        categories=VALID_CATEGORIES,
        active_page="upload"
    )

@app.route("/analysis")
@login_required
def analysis():
    df = load_data()
    category = request.args.get("category", "All Categories")
    sentiment = request.args.get("sentiment", "All Sentiments")
    search = request.args.get("search", "")

    filtered = apply_filters(df, category, sentiment, search)

    try:
        page = max(1, int(request.args.get("page", 1)))
    except ValueError:
        page = 1

    per_page = 15
    total_pages = max(1, math.ceil(len(filtered) / per_page))
    page = min(page, total_pages)
    start = (page - 1) * per_page

    rows = filtered.iloc[start:start + per_page].to_dict("records")
    metadata = load_metadata()

    return render_template(
        "analysis.html",
        rows=rows,
        total=len(filtered),
        page=page,
        total_pages=total_pages,
        category=category,
        sentiment=sentiment,
        search=search,
        categories=VALID_CATEGORIES,
        sentiments=VALID_SENTIMENTS,
        metadata=metadata,
        active_page="analysis"
    )

@app.route("/reports")
@login_required
def reports():
    df = load_data()
    data = get_dashboard_data(df)
    metadata = load_metadata()

    return render_template(
        "reports.html",
        data=data,
        metadata=metadata,
        categories=VALID_CATEGORIES,
        active_page="reports"
    )

@app.route("/download-report")
@login_required
def download_report():
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
        )
    except ImportError:
        flash("ReportLab is not installed. Run: pip install reportlab", "danger")
        return redirect(url_for("reports"))

    df = load_data()
    metadata = load_metadata()

    if df.empty:
        flash("There is no analysed feedback available to generate a report.", "warning")
        return redirect(url_for("reports"))

    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=35,
        leftMargin=35,
        topMargin=35,
        bottomMargin=35
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Title"], alignment=TA_CENTER,
        fontSize=18, leading=22, spaceAfter=8
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle", parent=styles["Normal"], alignment=TA_CENTER,
        fontSize=10, leading=14, spaceAfter=15
    )
    heading_style = ParagraphStyle(
        "ReportHeading", parent=styles["Heading2"],
        fontSize=13, leading=16, spaceBefore=12, spaceAfter=8
    )
    normal_style = ParagraphStyle(
        "ReportNormal", parent=styles["Normal"], fontSize=9, leading=12
    )
    small_style = ParagraphStyle(
        "ReportSmall", parent=styles["Normal"], fontSize=7, leading=9
    )
    table_header_style = ParagraphStyle(
        "TableHeader", parent=styles["Normal"], fontSize=8,
        leading=10, textColor=colors.white
    )

    elements = [
        Paragraph("STUDENT FEEDBACK ANALYSIS SYSTEM", title_style),
        Paragraph("Sentiment Analysis Report", subtitle_style),
        Paragraph("Generated: " + datetime.now().strftime("%d %B %Y, %H:%M:%S"), normal_style),
        Spacer(1, 12)
    ]

    elements.append(Paragraph("1. Analysis Summary", heading_style))
    total_records = len(df)
    sentiment_series = df["Sentiment"].astype(str).str.strip().str.lower()
    positive_count = int((sentiment_series == "positive").sum())
    neutral_count = int((sentiment_series == "neutral").sum())
    negative_count = int((sentiment_series == "negative").sum())

    def pct(count, total):
        return (count / total * 100) if total else 0.0

    summary_data = [
        [Paragraph("<b>Metric</b>", table_header_style), Paragraph("<b>Value</b>", table_header_style)],
        ["Total Feedback", str(total_records)],
        ["Positive Feedback", str(positive_count)],
        ["Neutral Feedback", str(neutral_count)],
        ["Negative Feedback", str(negative_count)]
    ]
    summary_table = Table(summary_data, colWidths=[250, 150])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563EB")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F4F6")]),
        ("PADDING", (0, 0), (-1, -1), 7)
    ]))
    elements.append(summary_table)

    elements.append(Paragraph("2. Sentiment Distribution", heading_style))
    sentiment_data = [
        [Paragraph("<b>Sentiment</b>", table_header_style), Paragraph("<b>Count</b>", table_header_style), Paragraph("<b>Percentage</b>", table_header_style)],
        ["Positive", str(positive_count), f"{pct(positive_count, total_records):.1f}%"],
        ["Neutral", str(neutral_count), f"{pct(neutral_count, total_records):.1f}%"],
        ["Negative", str(negative_count), f"{pct(negative_count, total_records):.1f}%"]
    ]
    sentiment_table = Table(sentiment_data, colWidths=[180, 100, 120])
    sentiment_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563EB")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("PADDING", (0, 0), (-1, -1), 7),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F4F6")])
    ]))
    elements.append(sentiment_table)

    elements.append(Paragraph("3. Model Performance", heading_style))
    metrics = metadata.get("metrics", {})
    accuracy = float(metrics.get("accuracy", 0))
    precision = float(metrics.get("precision", 0))
    recall = float(metrics.get("recall", 0))
    f1_score = float(metrics.get("f1", 0))
    cv_accuracy = float(metrics.get("cv_mean_accuracy", 0))

    model_data = [
        [Paragraph("<b>Metric</b>", table_header_style), Paragraph("<b>Value</b>", table_header_style)],
        ["Accuracy", f"{accuracy * 100:.2f}%"],
        ["Precision", f"{precision * 100:.2f}%"],
        ["Recall", f"{recall * 100:.2f}%"],
        ["F1-score", f"{f1_score * 100:.2f}%"],
        ["5-Fold Cross-Validation Accuracy", f"{cv_accuracy * 100:.2f}%"]
    ]
    model_table = Table(model_data, colWidths=[280, 120])
    model_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563EB")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("PADDING", (0, 0), (-1, -1), 7),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F4F6")])
    ]))
    elements.append(model_table)

    elements.append(Paragraph("4. Category Sentiment Comparison", heading_style))
    category_rows = [[
        Paragraph("<b>Category</b>", table_header_style),
        Paragraph("<b>Positive</b>", table_header_style),
        Paragraph("<b>Neutral</b>", table_header_style),
        Paragraph("<b>Negative</b>", table_header_style)
    ]]

    for category in VALID_CATEGORIES:
        category_df = df[df["Category"] == category]
        cs = category_df["Sentiment"].astype(str).str.strip().str.lower()
        category_rows.append([
            category,
            str(int((cs == "positive").sum())),
            str(int((cs == "neutral").sum())),
            str(int((cs == "negative").sum()))
        ])

    category_table = Table(category_rows, colWidths=[220, 70, 70, 70])
    category_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563EB")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F4F6")])
    ]))
    elements.append(category_table)

    elements.append(PageBreak())
    elements.append(Paragraph("5. Detailed Feedback Results", heading_style))
    feedback_rows = [[
        Paragraph("<b>ID</b>", table_header_style),
        Paragraph("<b>Feedback Text</b>", table_header_style),
        Paragraph("<b>Category</b>", table_header_style),
        Paragraph("<b>Sentiment</b>", table_header_style),
        Paragraph("<b>Date</b>", table_header_style)
    ]]

    for _, row in df.iterrows():
        feedback_text = str(row.get("Feedback Text", ""))
        if len(feedback_text) > 150:
            feedback_text = feedback_text[:147] + "..."

        feedback_rows.append([
            str(row.get("ID", "")),
            Paragraph(feedback_text, small_style),
            Paragraph(str(row.get("Category", "")), small_style),
            Paragraph(str(row.get("Sentiment", "")), small_style),
            str(row.get("Date", ""))
        ])

    feedback_table = Table(feedback_rows, colWidths=[35, 205, 105, 65, 70], repeatRows=1)
    feedback_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563EB")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")])
    ]))
    elements.append(feedback_table)
    elements.append(Spacer(1, 15))
    elements.append(Paragraph("Generated by the Student Feedback Analysis System.", normal_style))

    document.build(elements)
    buffer.seek(0)

    res = send_file(
        buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name="student_feedback_analysis_report.pdf"
    )
    res.headers["Content-Type"] = "application/pdf"
    res.headers["Content-Disposition"] = 'attachment; filename="student_feedback_analysis_report.pdf"'
    res.headers["Cache-Control"] = "no-store"
    return res

@app.route("/settings")
@login_required
def settings():
    metadata = load_metadata()
    return render_template(
        "settings.html",
        metadata=metadata,
        categories=VALID_CATEGORIES,
        active_page="settings"
    )

@app.route("/api/dashboard")
def api_dashboard():
    return jsonify(get_dashboard_data())

@app.errorhandler(413)
def too_large(_):
    flash("The uploaded file exceeds the 10 MB limit.", "danger")
    return redirect(url_for("upload"))

if __name__ == "__main__":
    app.run(debug=True)