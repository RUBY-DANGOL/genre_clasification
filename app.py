from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import io
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the model
model = load_model('music_genre_hrnet_best.h5')

# Define genre labels (adjust these based on your model's output)
GENRES = ['bhojpuri', 'dohori', 'newari', 'rock', 'tamangselo']

def create_spectrogram(audio_path):
    """
    Create a mel spectrogram from an audio file
    Returns the spectrogram as a numpy array
    """
    # Load audio file
    y, sr = librosa.load(audio_path, duration=30)
    
    # Create mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    return mel_spectrogram_db

def get_spectrogram_image(mel_spectrogram_db):
    """
    Convert mel spectrogram to base64 encoded image for display
    """
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spectrogram_db, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    
    # Save to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    # Encode to base64
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{image_base64}"

def preprocess_spectrogram(mel_spectrogram_db):
    """
    Preprocess the spectrogram for model input
    Adjust this based on your model's expected input shape
    """
    # Resize or pad to fixed shape - your model expects (128, 128, 1)
    target_shape = (128, 128)
    
    if mel_spectrogram_db.shape[1] < target_shape[1]:
        # Pad if shorter
        pad_width = target_shape[1] - mel_spectrogram_db.shape[1]
        mel_spectrogram_db = np.pad(mel_spectrogram_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
        # Truncate if longer
        mel_spectrogram_db = mel_spectrogram_db[:, :target_shape[1]]
    
    # Ensure we have exactly 128 mel bands
    if mel_spectrogram_db.shape[0] < target_shape[0]:
        pad_width = target_shape[0] - mel_spectrogram_db.shape[0]
        mel_spectrogram_db = np.pad(mel_spectrogram_db, ((0, pad_width), (0, 0)), mode='constant')
    elif mel_spectrogram_db.shape[0] > target_shape[0]:
        mel_spectrogram_db = mel_spectrogram_db[:target_shape[0], :]
    
    # Normalize - handle cases where std is 0 or results in NaN
    mean_val = mel_spectrogram_db.mean()
    std_val = mel_spectrogram_db.std()
    
    if std_val > 0 and not np.isnan(std_val):
        mel_spectrogram_db = (mel_spectrogram_db - mean_val) / std_val
    else:
        # If std is 0 or NaN, just center the data
        mel_spectrogram_db = mel_spectrogram_db - mean_val
    
    # Replace any remaining NaN or inf values with 0
    mel_spectrogram_db = np.nan_to_num(mel_spectrogram_db, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Add batch and channel dimensions: (128, 128) -> (1, 128, 128, 1)
    mel_spectrogram_db = np.expand_dims(mel_spectrogram_db, axis=0)
    mel_spectrogram_db = np.expand_dims(mel_spectrogram_db, axis=-1)
    
    return mel_spectrogram_db

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        
        if audio_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file
        filename = 'temp_audio.wav'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        audio_file.save(filepath)
        
        # Create spectrogram
        mel_spectrogram_db = create_spectrogram(filepath)
        
        # Get spectrogram image for display
        spectrogram_image = get_spectrogram_image(mel_spectrogram_db)
        
        # Preprocess for model
        processed_spectrogram = preprocess_spectrogram(mel_spectrogram_db)
        
        # Make prediction
        predictions = model.predict(processed_spectrogram)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = [
            {
                'genre': GENRES[i] if i < len(GENRES) else f'Genre {i}',
                'confidence': float(predictions[0][i]) * 100
            }
            for i in top_3_indices
        ]
        
        # Clean up
        os.remove(filepath)
        
        return jsonify({
            'genre': GENRES[predicted_class] if predicted_class < len(GENRES) else f'Genre {predicted_class}',
            'confidence': confidence * 100,
            'top_predictions': top_3_predictions,
            'spectrogram': spectrogram_image
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
