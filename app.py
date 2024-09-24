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

# Función para aplicar VAD (detección de actividad de voz)
def apply_vad(audio, sr, frame_duration=30):
    vad = webrtcvad.Vad(3)  # Modo agresivo para VAD
    frames = librosa.util.frame(audio, frame_length=int(sr * frame_duration / 1000), hop_length=int(sr * frame_duration / 1000))
    speech_flags = []

    for frame in frames.T:
        frame_bytes = (frame * 32768).astype(np.int16).tobytes()
        speech_flags.append(vad.is_speech(frame_bytes, sr))

    return np.array(speech_flags)

# Función para realizar la diarización de audio usando clustering
def diarize_audio(audio, sr, num_speakers=2):
    speech_flags = apply_vad(audio, sr)
    speech_indices = np.where(speech_flags == True)[0]
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13).T[speech_indices]
    clustering = AgglomerativeClustering(n_clusters=num_speakers, metric='cosine', linkage='average')
    speaker_labels = clustering.fit_predict(mfccs)
    return speaker_labels, speech_indices

# Función para transcribir audio usando Whisper con una barra de progreso
def transcribe_audio_data_with_progress(audio_data, sr):
    model = whisper.load_model("base")
    if sr != 16000:
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
    
    progress_bar = st.progress(0)
    total_duration = len(audio_data) / sr  # Duración total del audio en segundos
    result = {'segments': []}
    segments = model.transcribe(audio_data, verbose=False, fp16=False)['segments']
    
    for i, segment in enumerate(segments):
        result['segments'].append(segment)
        progress = (segment['end'] / total_duration)  # Actualizar progreso
        progress_bar.progress(min(progress, 1.0))
        time.sleep(0.1)  # Pequeña pausa para simular el progreso en tiempo real
    
    return result

# Función para crear el prompt para GPT
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

    Aquí está la transcripción etiquetada:

    {conversation}
    """
    return prompt

# Función para enviar el prompt a GPT-4o Mini y obtener el análisis
def analyze_call_with_gpt_mini(prompt, api_key):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "Eres un asistente útil."},
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

# Función para analizar una sola llamada
def analyze_single_call(audio_file, api_key):
    audio_data, sr = librosa.load(audio_file, sr=None)
    speaker_labels, speech_indices = diarize_audio(audio_data, sr)
    result = transcribe_audio_data_with_progress(audio_data, sr)

    transcript_with_speakers = []
    for segment in result['segments']:
        word_start = segment['start']
        nearest_index = np.argmin(np.abs(speech_indices - (word_start * sr)))
        speaker = speaker_labels[nearest_index]
        transcript_with_speakers.append(f"Speaker {speaker}: {segment['text']}")

    prompt = create_prompt(transcript_with_speakers)
    analysis = analyze_call_with_gpt_mini(prompt, api_key)

    return transcript_with_speakers, analysis

# Función para analizar múltiples llamadas (ZIP)
def analyze_multiple_calls(zip_file, api_key):
    results = []
    
    with zipfile.ZipFile(zip_file, 'r') as z:
        for audio_filename in z.namelist():
            with z.open(audio_filename) as audio_file:
                transcript, analysis = analyze_single_call(audio_file, api_key)
                results.append({
                    "Llamada": audio_filename,
                    "Transcripción": "Click para ver transcripción (desplegable)",
                    "Tipo de llamada": "",  # Llenado por el análisis GPT
                    "Razón": "",  # Llenado por el análisis GPT
                    "Información solicitada": "",  # Llenado por el análisis GPT
                    "Resolución de la llamada": "",  # Llenado por el análisis GPT
                    "Sentimiento": "",  # Llenado por el análisis GPT
                    "Observaciones": ""  # Llenado por el análisis GPT
                })
    return results

# Interfaz de Streamlit en español
st.title("Herramienta de análisis de llamadas")

# Sidebar para ingresar la API Key
api_key = st.sidebar.text_input("Introduce tu OpenAI API Key", type="password")

# Opción para seleccionar entre análisis de una llamada o varias
analysis_type = st.radio("Selecciona tipo de análisis", ("Análisis de una llamada", "Análisis de varias llamadas (ZIP)"))

if api_key:
    if analysis_type == "Análisis de una llamada":
        uploaded_file = st.file_uploader("Sube un archivo de audio", type=["mp3", "wav"])
        
        if uploaded_file:
            st.audio(uploaded_file, format="audio/mp3")
            st.write("Transcribiendo el audio... Por favor espera.")
            transcript, analysis = analyze_single_call(uploaded_file, api_key)
            
            # Mostrar transcripción en un desplegable
            with st.expander("Mostrar llamada transcrita"):
                for line in transcript:
                    st.write(line)
            
            # Mostrar resultado del análisis
            st.write("\nResultado del análisis:")
            st.write(analysis)
    
    elif analysis_type == "Análisis de varias llamadas (ZIP)":
        uploaded_zip = st.file_uploader("Sube un archivo ZIP con varios audios", type=["zip"])
        
        if uploaded_zip:
            st.write("Procesando el archivo ZIP... Por favor espera.")
            results = analyze_multiple_calls(uploaded_zip, api_key)
            
            # Mostrar resultados en tabla
            df = pd.DataFrame(results)
            st.write(df)

            # Descargar resultados como CSV
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(label="Descargar resultados como CSV", data=csv, file_name='resultados_analisis_llamadas.csv', mime='text/csv')
