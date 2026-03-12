from flask import Flask, render_template, request, jsonify
import os
import shutil
from processor import ingest_documents, agent_dispatcher

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure clean start
if os.path.exists(UPLOAD_FOLDER):
    shutil.rmtree(UPLOAD_FOLDER)
os.makedirs(UPLOAD_FOLDER)

current_db = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    files = request.files.getlist('files')
    for file in files:
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
    return jsonify({"message": f"{len(files)} files uploaded to staging. Ready to process."})

@app.route('/process', methods=['POST'])
def process():
    global current_db
    # Process everything in the uploads folder at once
    current_db = ingest_documents(app.config['UPLOAD_FOLDER'])
    if current_db:
        return jsonify({"message": "Knowledge base created successfully!"})
    return jsonify({"message": "No valid files found."}), 400

@app.route('/ask', methods=['POST'])
def ask():
    if not current_db:
        return jsonify({"answer": "Please process documents first!"})
    query = request.json.get('query')
    response = agent_dispatcher(query, current_db)
    return jsonify({"answer": response})

if __name__ == '__main__':
    app.run(debug=True)