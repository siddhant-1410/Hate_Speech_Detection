# Hate Speech Detection Chatbot

A hate speech detection system with an interactive chatbot interface built using Deep Learning, NLP and Streamlit. This project classifies text into three categories: Hate Speech, Offensive Language, and Neither.

## Overview

This project implements a hate speech detection system that can analyze text content and classify it into different categories. The system uses a deep learning model (LSTM) trained on text data and provides a chatbot-style interface for users to interact with the model.

## Features

- Interactive chatbot interface for natural user interaction
- Classification of text input
- Confidence scoring with visual indicators
- Color-coded results for different classification types
- Chat history maintained during sessions
- Simple and intuitive user interface

## Classification Categories

- **Hate Speech**: Contains hateful content targeting individuals or groups
- **Offensive Language**: Offensive but not necessarily hateful
- **Neither**: Clean, neutral content

## Installation

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/hate-speech-detection-chatbot.git
   cd hate-speech-detection-chatbot
   ```

2. Create a virtual environment
   ```bash
   python -m venv your_virtual_envrionment_name
   ```

3. Install required packages
   ```bash
   pip install -r requirements.txt
   ```

4. Download NLTK data
   ```bash
   python download_nltk_data.py
   ```

## Usage

1. Start the Streamlit app
   ```bash
   streamlit run app.py
   ```

2. Open your browser and navigate to `http://localhost:8501`

3. Start chatting with the hate speech detection bot

## Technical Details

### Model Architecture
- Type: Long Short-Term Memory (LSTM) Neural Network
- Framework: TensorFlow/Keras
- Input: Tokenized and padded text sequences
- Output: 3-class probability distribution

### Text Preprocessing Pipeline
1. User mention normalization (replace @mentions with "user")
2. HTML entity removal
3. URL removal
4. Noise removal (special characters and punctuation)
5. Tokenization
6. Stopword removal
7. Lemmatization

### Key Libraries
- Streamlit: Web application framework
- TensorFlow/Keras: Deep learning model
- NLTK: Natural language processing
- NumPy: Numerical computations
- Pickle: Model serialization

## Requirements

- Python 3.7+
- TensorFlow
- Streamlit
- NLTK
- NumPy
- Pickle

## Troubleshooting

**NLTK Resource Error:**
```
LookupError: Resource punkt_tab not found
```
Solution: Run `python download_nltk_data.py` to download required NLTK data.

**Model Loading Error:**
```
FileNotFoundError: No such file or directory: 'model/best_lstm_model.h5'
```
Solution: Ensure all model files are in the `model/` directory.

## License

This project is licensed under the MIT License.
