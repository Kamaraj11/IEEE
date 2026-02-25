import os
import sys
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

# Add project root to sys path so we can import our modules correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model_loader import ModelLoader
from report_generator import ReportGenerator
from progress_tracker import ProgressTracker

app = Flask(__name__)
app.secret_key = 'ieee_derm_pipeline_secret'

# Setup upload folder to save images temporarily
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# Secure file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize singletons on startup
print("Initializing AI Pipeline...")
try:
    # Look for model from project root
    model_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), "ham10000_model.pth")
    predictor = ModelLoader(model_path=model_path)
except Exception as e:
    print(f"Warning: Could not spin up model. Reason: {e}")
    predictor = None
    
report_generator = ReportGenerator()
tracker = ProgressTracker()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def run_prediction():
    if predictor is None:
        flash('Model failed to load on backend.', 'error')
        return redirect(url_for('index'))

    # Check if a file was uploaded
    if 'file' not in request.files:
        flash('No file detected', 'error')
        return redirect(url_for('index'))
        
    file = request.files['file']
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('index'))

    # Check if there is an old probability for follow-up tracking
    old_prob_str = request.form.get('old_prob', '')
    old_prob = float(old_prob_str) if old_prob_str else None

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # 1. Run model prediction
        result = predictor.predict(file_path)
        
        if result is None:
             flash('Error processing image through model.', 'error')
             return redirect(url_for('index'))
             
        # 2. Extract results & Handle Gates
        if "prediction" in result:
            # Caught by Gate / Rejection Layer
            pred_class = result["prediction"]
            probability = result["confidence"]
            medical_report = result["medical_report"]
            short_name = "normal" if "No clear lesion" in pred_class else "uncertain"
        else:
            # Std Classifier Output
            pred_class = result["class_name"]
            short_name = result["short_name"]
            probability = result["probability"]
            
            # 3. Generate detailed IEEE-aligned report
            medical_report = report_generator.generate(short_name)
        
        # 4. Progress tracker logic (Follow-ups) 
        progress_alert = None
        if old_prob is not None:
            # We track the change in the *specific* top predicted class today, against what it was previously
            progress_alert = tracker.evaluate_progress(old_prob, probability, short_name)

        return render_template('result.html', 
                               image_file=filename,
                               pred_class=pred_class,
                               probability=probability,
                               report=medical_report,
                               progress=progress_alert,
                               short_name=short_name)
    
    flash('File type not allowed. Please upload JPG or PNG.', 'error')
    return redirect(url_for('index'))

if __name__ == '__main__':
    # Clean output formatting explicitly requesting Port 5001 
    print("🚀 Starting Flask Server on http://0.0.0.0:5001")
    app.run(host='0.0.0.0', port=5001, debug=False)
