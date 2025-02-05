from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from enhance import load_model, process_file, process_video

app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov', 'mkv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/enhance', methods=['POST'])
def enhance():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        file.save(input_path)

        # Load model
        model_choice = request.form.get('model', 'x2')
        use_gpu = request.form.get('use_gpu', 'false').lower() == 'true'
        use_cpu_egpu = request.form.get('use_cpu_egpu', 'false').lower() == 'true'
        sr = load_model(model_choice, use_gpu, use_cpu_egpu)

        # Process file
        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            success = process_video(input_path, output_path, sr)
        else:
            success = process_file(input_path, output_path, sr)

        if success:
            return jsonify({'message': 'File enhanced successfully', 'output_path': output_path}), 200
        else:
            return jsonify({'error': 'Enhancement failed'}), 500

    return jsonify({'error': 'File type not allowed'}), 400

if __name__ == '__main__':
    app.run(debug=True) 