import streamlit as st
import librosa
import whisper
import numpy as np
import zipfile
from sklearn.cluster import AgglomerativeClustering
import webrtcvad
import requests
import tempfile
import os
import pandas as pd
import time

# Function to perform Voice Activity Detection (VAD) using WebRTC VAD
def apply_vad(audio, sr, frame_duration=30):
    vad = webrtcvad.Vad(3)  # Aggressive mode for VAD
    frames = librosa.util.frame(audio, frame_length=int(sr * frame_duration / 1000), hop_length=int(sr * frame_duration / 1000))
    speech_flags = []

    for frame in frames.T:
        frame_bytes = (frame * 32768).astype(np.int16).tobytes()
        speech_flags.append(vad.is_speech(frame_bytes, sr))

    return np.array(speech_flags)

# Function to extract features and perform diarization using spectral clustering
def diarize_audio(audio, sr, num_speakers=2):
    # Apply Voice Activity Detection (VAD)
    speech_flags = apply_vad(audio, sr)
    speech_indices = np.where(speech_flags == True)[0]

    # Extract MFCCs (Mel Frequency Cepstral Coefficients) for speaker features
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13).T[speech_indices]

    # Apply Spectral Clustering for speaker diarization
    clustering = AgglomerativeClustering(n_clusters=num_speakers, metric='cosine', linkage='average')
    speaker_labels = clustering.fit_predict(mfccs)

    return speaker_labels, speech_indices

# Whisper transcription function using audio data directly (bypassing ffmpeg)
def transcribe_audio_data_with_progress(audio_data, sr):
    model = whisper.load_model("base")
    
    # Whisper expects 16kHz audio, so resample if needed
    if sr != 16000:
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
    
    # Initialize progress bar
    progress_bar = st.progress(0)
    total_duration = len(audio_data) / sr  # Total duration of the audio in seconds
    
    # Perform transcription with progress
    result = {'segments': []}
    segments = model.transcribe(audio_data, verbose=False, fp16=False)['segments']
    
    for i, segment in enumerate(segments):
        result['segments'].append(segment)
        progress = (segment['end'] / total_duration)  # Update progress
        progress_bar.progress(min(progress, 1.0))  # Ensure it doesn't go above 1.0
        time.sleep(0.1)  # Small delay to simulate real-time progress
    
    return result

# Function to create the prompt for GPT-4o Mini analysis
def create_prompt(transcription):
    conversation = "\n".join(transcription)
    prompt = f"""
    A continuación tienes una conversación entre un operador de servicio al cliente (Speaker 0) y un cliente (Speaker 1).
    Necesito que dividas la relación de llamadas en los siguientes tipos:

    1. Informativas (horarios, CLUB, disponibilidad o características de productos, financiación, etc.)
    2. Sobre estado pedidos (no entregados, retrasados, cambio de dirección de entrega o recogida, o método de entrega/recogida)
    3. Reclamaciones sobre productos/pedidos ya entregados (mala experiencia, no funciona, está roto, quiero devolverlo,…)
    4. Intención de compra (quieren comprar un producto o contratar un servicio del cual no existe ningún pedido previo)
    5. Otras.

    También hay que identificar el sentimiento de cada una de estas tipologías (positivo, negativo, neutro).

    En relación a las llamadas informativas, necesitamos saber cuáles son las más repetidas y qué tipo de información se solicita.
    Para pedidos, necesitamos identificar si hay alguna tienda, tipo de producto o servicio que tenga más índice de incidencias o reclamaciones.

    Además, necesitamos identificar si la llamada ha quedado resuelta o no.

    Aquí está la transcripción etiquetada:

    {conversation}
    """
    return prompt

# Function to send the prompt to GPT-4-0 Mini and get the analysis
def analyze_call_with_gpt_mini(prompt, api_key):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 1500
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error {response.status_code}: {response.text}"

# Function to analyze a single audio file
def analyze_single_call(audio_file, api_key):
    # Process the audio
    audio_data, sr = librosa.load(audio_file, sr=None)
    speaker_labels, speech_indices = diarize_audio(audio_data, sr)
    result = transcribe_audio_data_with_progress(audio_data, sr)

    # Generate the transcript with speaker labels
    transcript_with_speakers = []
    for segment in result['segments']:
        word_start = segment['start']
        nearest_index = np.argmin(np.abs(speech_indices - (word_start * sr)))
        speaker = speaker_labels[nearest_index]
        transcript_with_speakers.append(f"Speaker {speaker}: {segment['text']}")

    # Create the prompt for GPT analysis
    prompt = create_prompt(transcript_with_speakers)
    analysis = analyze_call_with_gpt_mini(prompt, api_key)

    return transcript_with_speakers, analysis

# Function to handle multiple audio files (ZIP)
def analyze_multiple_calls(zip_file, api_key):
    results = []
    
    # Unzip and process each file
    with zipfile.ZipFile(zip_file, 'r') as z:
        for audio_filename in z.namelist():
            with z.open(audio_filename) as audio_file:
                transcript, analysis = analyze_single_call(audio_file, api_key)
                
                # Store results for each call
                results.append({
                    "Llamada": audio_filename,
                    "Transcripción": "Click para ver transcripción (desplegable)",
                    "Tipo de llamada": "",  # This would be filled by the GPT analysis
                    "Razón": "",  # Filled by analysis
                    "Información solicitada": "",  # Filled by analysis
                    "Resolución de la llamada": "",  # Filled by analysis
                    "Sentimiento": "",  # Filled by analysis
                    "Observaciones": ""  # Filled by analysis
                })
    return results

# Streamlit interface
st.title("Call Analysis Tool")

# Sidebar for API key input
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

# Option to choose single or multiple call analysis
analysis_type = st.radio("Select analysis type", ("Single Call", "Multiple Calls (ZIP)"))

if api_key:
    if analysis_type == "Single Call":
        uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])
        
        if uploaded_file:
            st.audio(uploaded_file, format="audio/mp3")
            st.write("Transcribing the audio... Please wait.")
            transcript, analysis = analyze_single_call(uploaded_file, api_key)
            
            # Display the transcription inside an expander (collapsible section)
            with st.expander("Mostrar llamada transcrita"):
                for line in transcript:
                    st.write(line)
            
            # Display analysis result
            st.write("\nAnalysis Result:")
            st.write(analysis)
    
    elif analysis_type == "Multiple Calls (ZIP)":
        uploaded_zip = st.file_uploader("Upload a ZIP file with audio files", type=["zip"])
        
        if uploaded_zip:
            st.write("Processing the ZIP file... Please wait.")
            results = analyze_multiple_calls(uploaded_zip, api_key)
            
            # Display results in a table
            df = pd.DataFrame(results)
            st.write(df)

            # Allow user to download the results
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download results as CSV", data=csv, file_name='call_analysis_results.csv', mime='text/csv')
