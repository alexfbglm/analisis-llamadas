import streamlit as st
import librosa
import whisper
import numpy as np
import zipfile
from sklearn.cluster import AgglomerativeClustering
import webrtcvad
import requests
import pandas as pd
import json
from scipy.spatial.distance import pdist, squareform
import tempfile
import os
import time
from io import BytesIO

# Inicializar el estado para los análisis, archivos procesados y transcripciones
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'analysis_results' not in st.session_state:
    st.session_state['analysis_results'] = []
if 'processed_files' not in st.session_state:
    st.session_state['processed_files'] = []
if 'transcriptions' not in st.session_state:
    st.session_state['transcriptions'] = {}

# Función para aplicar VAD (detección de actividad de voz)
def apply_vad(audio, sr, frame_duration=30):
    vad = webrtcvad.Vad(3)  # Modo agresivo para VAD
    frame_length = int(sr * frame_duration / 1000)
    frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=frame_length).T
    speech_flags = []

    for frame in frames:
        frame_bytes = (frame * 32768).astype(np.int16).tobytes()
        speech_flags.append(vad.is_speech(frame_bytes, sr))

    return np.array(speech_flags)

# Función para realizar la diarización de audio usando clustering
def diarize_audio(audio, sr, num_speakers=2):
    speech_flags = apply_vad(audio, sr)
    speech_indices = np.where(speech_flags == True)[0]
    
    if len(speech_indices) == 0:
        return np.array([]), np.array([])
    
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13).T[speech_indices]

    # Calculamos la matriz de distancias utilizando la distancia de coseno
    distance_matrix = squareform(pdist(mfccs, metric='cosine'))

    # Aplicamos AgglomerativeClustering usando la distancia precomputada
    clustering = AgglomerativeClustering(n_clusters=num_speakers, metric='precomputed', linkage='average')
    speaker_labels = clustering.fit_predict(distance_matrix)

    return speaker_labels, speech_indices

# Función para transcribir audio usando Whisper con barra de progreso
def transcribe_audio_data_with_progress(audio_data, sr):
    model = whisper.load_model("base")
    
    # Whisper espera audio a 16kHz, resample si es necesario
    if sr != 16000:
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
    
    # Inicializar barra de progreso
    progress_bar = st.progress(0)
    total_segments = 0
    try:
        segments = model.transcribe(audio_data, verbose=False, fp16=False)['segments']
        total_segments = len(segments)
    except Exception as e:
        st.error(f"Error al transcribir el audio: {e}")
        return {'segments': []}
    
    # Realizar transcripción con actualización de progreso
    result = {'segments': []}
    for i, segment in enumerate(segments):
        result['segments'].append(segment)
        progress = (i + 1) / total_segments if total_segments > 0 else 1
        progress_bar.progress(min(progress, 1.0))  # Asegurar que no exceda 1.0
        time.sleep(0.05)  # Pequeña demora para simular progreso en tiempo real
    
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

    Por favor, devuelve el análisis en el siguiente formato JSON **sin ningún bloque de código**:
    {{
        "tipo_llamada": "Tipo de llamada",
        "razon": "Razón de la llamada",
        "info_solicitada": "Información solicitada",
        "resolucion": "Resolución de la llamada",
        "sentimiento": "Sentimiento detectado",
        "observaciones": "Observaciones adicionales"
    }}

    Aquí está la transcripción etiquetada:

    {conversation}
    """
    return prompt

# Función para enviar el prompt a GPT-4 y obtener el análisis en formato JSON
def analyze_call_with_gpt_mini(prompt, api_key):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "gpt-4o-mini",  # Asegúrate de que el modelo sea correcto
        "messages": [
            {"role": "system", "content": "Eres un asistente útil."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 1500
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        analysis = response.json()["choices"][0]["message"]["content"]
        # Intentamos convertir la respuesta a JSON
        analysis_json = json.loads(analysis)
        return analysis_json
    except requests.exceptions.RequestException as e:
        return f"Error en la solicitud: {e}"
    except json.JSONDecodeError:
        return f"Error al parsear el JSON: {analysis}"

# Función para analizar una sola llamada
def analyze_single_call(audio_path, api_key):
    try:
        audio_data, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        st.error(f"Error al cargar el archivo de audio {os.path.basename(audio_path)}: {e}")
        return None

    speaker_labels, speech_indices = diarize_audio(audio_data, sr)
    if len(speaker_labels) == 0:
        st.warning(f"No se detectó actividad de voz en {os.path.basename(audio_path)}.")
    
    result = transcribe_audio_data_with_progress(audio_data, sr)

    if not result['segments']:
        st.warning(f"No se pudieron transcribir segmentos en {os.path.basename(audio_path)}.")
        return None

    transcript_with_speakers = []
    for segment in result['segments']:
        word_start = segment['start']
        if len(speech_indices) == 0:
            speaker = "Desconocido"
        else:
            nearest_index = np.argmin(np.abs(speech_indices - int(word_start * sr)))
            if len(speaker_labels) > nearest_index:
                speaker = speaker_labels[nearest_index]
            else:
                speaker = "Desconocido"
        transcript_with_speakers.append(f"Speaker {speaker}: {segment['text']}")

    # Almacenar la transcripción en el estado
    st.session_state['transcriptions'][os.path.basename(audio_path)] = transcript_with_speakers

    prompt = create_prompt(transcript_with_speakers)
    analysis_json = analyze_call_with_gpt_mini(prompt, api_key)

    # Si no se pudo parsear a JSON, devolver el error como análisis
    if isinstance(analysis_json, str):
        return {
            "transcripcion": transcript_with_speakers,
            "error": analysis_json
        }

    # Si se parseó correctamente el JSON, extraer los campos
    return {
        "transcripcion": transcript_with_speakers,
        "tipo_llamada": analysis_json.get("tipo_llamada", ""),
        "razon": analysis_json.get("razon", ""),
        "info_solicitada": analysis_json.get("info_solicitada", ""),
        "resolucion": analysis_json.get("resolucion", ""),
        "sentimiento": analysis_json.get("sentimiento", ""),
        "observaciones": analysis_json.get("observaciones", "")
    }

# Función para analizar múltiples llamadas desde un archivo ZIP
def analyze_multiple_calls(zip_file, api_key):
    results = []
    
    with zipfile.ZipFile(zip_file, 'r') as z:
        audio_extensions = [".mp3", ".wav", ".m4a", ".flac", ".ogg"]
        audio_files = [f for f in z.namelist() if os.path.splitext(f)[1].lower() in audio_extensions]
        
        if not audio_files:
            st.error("El archivo ZIP no contiene archivos de audio compatibles.")
            return results
        
        st.success(f"Encontrados {len(audio_files)} archivos de audio para procesar.")
        
        for audio_filename in audio_files:
            # Normalizar el nombre del archivo (sin carpetas)
            normalized_filename = os.path.basename(audio_filename)
            
            if normalized_filename in st.session_state['processed_files']:
                st.info(f"{normalized_filename} ya ha sido procesado previamente. Mostrando resultados existentes.")
                # Encontrar el análisis correspondiente
                existing_analysis = next((item for item in st.session_state['analysis_results'] if item["Nombre de la llamada"] == normalized_filename), None)
                existing_transcription = st.session_state['transcriptions'].get(normalized_filename, [])
                if existing_analysis and existing_transcription:
                    with st.expander(f"Mostrar llamada transcrita: {normalized_filename}"):
                        for line in existing_transcription:
                            st.write(line)

                    with st.expander(f"Resultado del análisis: {normalized_filename}"):
                        st.json({
                            "Tipo de llamada": existing_analysis.get("tipo_llamada", ""),
                            "Razón": existing_analysis.get("razon", ""),
                            "Información solicitada": existing_analysis.get("info_solicitada", ""),
                            "Resolución de la llamada": existing_analysis.get("resolucion", ""),
                            "Sentimiento": existing_analysis.get("sentimiento", ""),
                            "Observaciones": existing_analysis.get("observaciones", "")
                        })
                results.append(existing_analysis)
                continue  # Saltar al siguiente archivo
            
            with z.open(audio_filename) as audio_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_filename)[1]) as temp_audio:
                    temp_audio.write(audio_file.read())
                    temp_audio_path = temp_audio.name

            st.write(f"### Procesando: {normalized_filename}")
            analysis = analyze_single_call(temp_audio_path, api_key)

            if analysis is None:
                st.warning(f"No se pudo procesar {normalized_filename}.")
                os.remove(temp_audio_path)
                continue

            # Mostrar la transcripción dentro de un expander
            with st.expander(f"Mostrar llamada transcrita: {normalized_filename}"):
                for line in analysis["transcripcion"]:
                    st.write(line)

            # Mostrar el resultado del análisis dentro de otro expander
            with st.expander(f"Resultado del análisis: {normalized_filename}"):
                if "error" in analysis:
                    st.error(analysis["error"])
                else:
                    st.json({
                        "Tipo de llamada": analysis["tipo_llamada"],
                        "Razón": analysis["razon"],
                        "Información solicitada": analysis["info_solicitada"],
                        "Resolución de la llamada": analysis["resolucion"],
                        "Sentimiento": analysis["sentimiento"],
                        "Observaciones": analysis["observaciones"]
                    })

            # Añadir los resultados al listado para generar el Excel
            result_entry = {
                "Nombre de la llamada": normalized_filename,
                "Tipo de llamada": analysis.get("tipo_llamada", ""),
                "Razón": analysis.get("razon", ""),
                "Información solicitada": analysis.get("info_solicitada", ""),
                "Resolución de la llamada": analysis.get("resolucion", ""),
                "Sentimiento": analysis.get("sentimiento", ""),
                "Observaciones": analysis.get("observaciones", "")
            }
            results.append(result_entry)
            st.session_state['analysis_results'].append(result_entry)
            st.session_state['processed_files'].append(normalized_filename)

            # Opcionalmente, permitir descargar la transcripción etiquetada
            transcript_text = "\n".join(analysis["transcripcion"])
            st.download_button(
                label=f"Descargar Transcripción Etiquetada: {normalized_filename}",
                data=transcript_text,
                file_name=f"labeled_transcript_{os.path.splitext(normalized_filename)[0]}.txt"
            )

            # Opcionalmente, permitir descargar el análisis en formato JSON
            if "tipo_llamada" in analysis:
                analysis_json_str = json.dumps({
                    "Tipo de llamada": analysis["tipo_llamada"],
                    "Razón": analysis["razon"],
                    "Información solicitada": analysis["info_solicitada"],
                    "Resolución de la llamada": analysis["resolucion"],
                    "Sentimiento": analysis["sentimiento"],
                    "Observaciones": analysis["observaciones"]
                }, ensure_ascii=False, indent=4)
                st.download_button(
                    label=f"Descargar Análisis JSON: {normalized_filename}",
                    data=analysis_json_str,
                    file_name=f"analysis_{os.path.splitext(normalized_filename)[0]}.json"
                )

            # Eliminar el archivo temporal después de procesarlo
            os.remove(temp_audio_path)
    
    return results

# Función para generar el archivo Excel
def generate_excel(results):
    df = pd.DataFrame(results)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Análisis de Llamadas')
    processed_data = output.getvalue()
    return processed_data

# Función para manejar el chat
def handle_chat(user_message, analysis_data, api_key):
    if not user_message:
        return ""
    
    # Preparar el contexto a partir de los datos de análisis
    context = "Aquí tienes los análisis de las llamadas:\n\n"
    for call in analysis_data:
        context += f"Llamada: {call['Nombre de la llamada']}\n"
        for key, value in call.items():
            if key != "Nombre de la llamada":
                context += f"{key}: {value}\n"
        context += "\n"
    
    prompt = f"""
    Usa la siguiente información de análisis de llamadas para responder a las preguntas del usuario.
    
    {context}
    
    Usuario: {user_message}
    Asistente:
    """
    
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
        "max_tokens": 500
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        assistant_reply = response.json()["choices"][0]["message"]["content"]
        return assistant_reply
    except requests.exceptions.RequestException as e:
        return f"Error en la solicitud: {e}"
    except json.JSONDecodeError:
        return "Error al procesar la respuesta del asistente."

# Interfaz de Streamlit en español
st.title("Herramienta de Análisis de Llamadas")

# Sidebar para ingresar la API Key y el Chat
with st.sidebar:
    st.header("Configuración")
    api_key = st.text_input("Introduce tu OpenAI API Key", type="password")
    
    st.header("Chat de Soporte")
    user_message = st.text_input("Escribe tu pregunta sobre los análisis de las llamadas:")
    if st.button("Enviar") and user_message:
        if st.session_state['analysis_results']:
            chat_response = handle_chat(user_message, st.session_state['analysis_results'], api_key)
            st.session_state['chat_history'].append({"usuario": user_message, "asistente": chat_response})
        else:
            st.warning("Por favor, realiza primero el análisis de las llamadas.")
    
    if st.session_state['chat_history']:
        st.subheader("Historial del Chat")
        for chat in reversed(st.session_state['chat_history']):
            st.markdown(f"**Usuario:** {chat['usuario']}")
            st.markdown(f"**Asistente:** {chat['asistente']}")

# Opción para seleccionar entre análisis de una llamada o varias
analysis_type = st.radio("Selecciona tipo de análisis", ("Análisis de una llamada", "Análisis de varias llamadas (ZIP)"))

if api_key:
    if analysis_type == "Análisis de una llamada":
        uploaded_file = st.file_uploader("Sube un archivo de audio", type=["mp3", "wav"])
        
        if uploaded_file:
            if uploaded_file.name not in st.session_state['processed_files']:
                st.audio(uploaded_file, format="audio/mp3")
                st.write("Transcribiendo y analizando el audio... Por favor espera.")
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_audio:
                    temp_audio.write(uploaded_file.read())
                    temp_audio_path = temp_audio.name
                
                analysis = analyze_single_call(temp_audio_path, api_key)
                
                # Eliminar el archivo temporal después de procesarlo
                os.remove(temp_audio_path)
                
                if analysis is not None:
                    # Mostrar transcripción en un desplegable
                    with st.expander("Mostrar llamada transcrita"):
                        for line in analysis["transcripcion"]:
                            st.write(line)
                    
                    # Mostrar resultado del análisis
                    with st.expander("Resultado del análisis"):
                        if "error" in analysis:
                            st.error(analysis["error"])
                        else:
                            st.json({
                                "Tipo de llamada": analysis["tipo_llamada"],
                                "Razón": analysis["razon"],
                                "Información solicitada": analysis["info_solicitada"],
                                "Resolución de la llamada": analysis["resolucion"],
                                "Sentimiento": analysis["sentimiento"],
                                "Observaciones": analysis["observaciones"]
                            })
                    
                    # Opcionalmente, permitir descargar la transcripción etiquetada
                    transcript_text = "\n".join(analysis["transcripcion"])
                    st.download_button(
                        label="Descargar Transcripción Etiquetada",
                        data=transcript_text,
                        file_name=f"labeled_transcript_{os.path.splitext(uploaded_file.name)[0]}.txt"
                    )
                    
                    # Opcionalmente, permitir descargar el análisis en formato JSON
                    if "tipo_llamada" in analysis:
                        analysis_json_str = json.dumps({
                            "Tipo de llamada": analysis["tipo_llamada"],
                            "Razón": analysis["razon"],
                            "Información solicitada": analysis["info_solicitada"],
                            "Resolución de la llamada": analysis["resolucion"],
                            "Sentimiento": analysis["sentimiento"],
                            "Observaciones": analysis["observaciones"]
                        }, ensure_ascii=False, indent=4)
                        st.download_button(
                            label="Descargar Análisis JSON",
                            data=analysis_json_str,
                            file_name=f"analysis_{os.path.splitext(uploaded_file.name)[0]}.json"
                        )
                    
                    # Guardar el análisis en el estado para el chat
                    st.session_state['analysis_results'].append({
                        "Nombre de la llamada": uploaded_file.name,
                        "Tipo de llamada": analysis.get("tipo_llamada", ""),
                        "Razón": analysis.get("razon", ""),
                        "Información solicitada": analysis.get("info_solicitada", ""),
                        "Resolución de la llamada": analysis.get("resolucion", ""),
                        "Sentimiento": analysis.get("sentimiento", ""),
                        "Observaciones": analysis.get("observaciones", "")
                    })
                    st.session_state['processed_files'].append(uploaded_file.name)
            else:
                st.info(f"El archivo {uploaded_file.name} ya ha sido procesado previamente.")
                # Mostrar los resultados existentes
                existing_analysis = next((item for item in st.session_state['analysis_results'] if item["Nombre de la llamada"] == uploaded_file.name), None)
                existing_transcription = st.session_state['transcriptions'].get(uploaded_file.name, [])
                if existing_analysis and existing_transcription:
                    with st.expander("Mostrar llamada transcrita"):
                        for line in existing_transcription:
                            st.write(line)
                    
                    with st.expander("Resultado del análisis"):
                        st.json({
                            "Tipo de llamada": existing_analysis.get("tipo_llamada", ""),
                            "Razón": existing_analysis.get("razon", ""),
                            "Información solicitada": existing_analysis.get("info_solicitada", ""),
                            "Resolución de la llamada": existing_analysis.get("resolucion", ""),
                            "Sentimiento": existing_analysis.get("sentimiento", ""),
                            "Observaciones": existing_analysis.get("observaciones", "")
                        })
    
    elif analysis_type == "Análisis de varias llamadas (ZIP)":
        uploaded_zip = st.file_uploader("Sube un archivo ZIP con varios audios", type=["zip"])
        
        if uploaded_zip:
            st.write(f"Archivo ZIP subido: {uploaded_zip.name}")
            st.write("Procesando el archivo ZIP... Por favor espera.")
            analysis_results = analyze_multiple_calls(uploaded_zip, api_key)
            
            # Guardar los análisis en el estado para el chat y para generar el Excel
            if analysis_results:
                st.session_state['analysis_results'].extend(analysis_results)
            
            # Generar y mostrar el botón para descargar el Excel
            if analysis_results:
                excel_data = generate_excel(analysis_results)
                st.download_button(
                    label="Descargar Análisis en Excel",
                    data=excel_data,
                    file_name="analisis_llamadas.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
else:
    st.warning("Por favor, introduce tu OpenAI API Key en la barra lateral.")
