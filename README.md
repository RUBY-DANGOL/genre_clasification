# Nepal Audio Classifier 🎵

It is a music genre classifer.



https://github.com/user-attachments/assets/e76f72c8-70f7-44bc-b011-1c82b761e24f


## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure your `music_genre_hrnet_best.h5` model file is in the root directory

## Usage

1. Run the Flask application:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Upload an audio file and click "Classify Genre" to see the results!

## Project Structure

```
audioclass/
├── app.py                 # Flask backend application
├── music_genre_hrnet_best.h5             # Pre-trained audio classification model
├── requirements.txt      # Python dependencies
├── templates/
│   └── index.html       # Main HTML template
├── static/
│   └── style.css        # Nepal-themed CSS styling
└── uploads/             # Temporary audio file storage (auto-created)
```

## Model Information

The application uses a pre-trained audio classification model (`music_genre_hrnet_best.h5`) that classifies audio into the following genres:
- bhojpuri
- dohori
- newari
- rock music
- tamang selo
  
**Note:** Adjust the `GENRES` list in `app.py` if your model uses different genre labels.

## Technical Details

- **Backend:** Flask (Python)
- **Frontend:** HTML5, CSS3, JavaScript
- **Audio Processing:** librosa
- **Deep Learning:** TensorFlow/Keras
- **Visualization:** matplotlib

## Customization

### Adjusting Model Input Shape

If your model expects a different input shape, modify the `preprocess_spectrogram()` function in `app.py`:

```python
target_shape = (128, 130)  # Change to match your model's expected input
```
