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

# ================================
# Inicialización del estado
# ================================
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'analysis_results' not in st.session_state:
    st.session_state['analysis_results'] = []
if 'processed_files' not in st.session_state:
    st.session_state['processed_files'] = []
if 'transcriptions' not in st.session_state:
    st.session_state['transcriptions'] = {}
if 'user_message' not in st.session_state:
    st.session_state['user_message'] = ''

# ================================
# Definición de Funciones
# ================================

def apply_vad(audio, sr, frame_duration=30):
    vad = webrtcvad.Vad(3)  # Modo agresivo para VAD
    frame_length = int(sr * frame_duration / 1000)
    frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=frame_length).T
    speech_flags = []

    for frame in frames:
        frame_bytes = (frame * 32768).astype(np.int16).tobytes()
        speech_flags.append(vad.is_speech(frame_bytes, sr))

    return np.array(speech_flags)

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

def analyze_call_with_gpt(prompt, api_key):
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

def handle_chat():
    user_message = st.session_state['user_message']
    if not user_message:
        return
    if st.session_state['analysis_results']:
        chat_response = get_chat_response(user_message, st.session_state['analysis_results'], st.session_state['api_key'])
        st.session_state['chat_history'].append({"usuario": user_message, "asistente": chat_response})
        st.session_state['user_message'] = ''  # Clear the input after processing
    else:
        st.warning("Por favor, realiza primero el análisis de las llamadas.")

def get_chat_response(user_message, analysis_data, api_key):
    if not user_message:
        return ""

    # Prepare context from analysis data
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

    # Make the API call
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
    analysis_json = analyze_call_with_gpt(prompt, api_key)

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

# Función para generar el archivo Excel
def generate_excel(results):
    df = pd.DataFrame(results)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Análisis de Llamadas')
    processed_data = output.getvalue()
    return processed_data

# ================================
# Interfaz de Usuario
# ================================

# Título de la aplicación
st.title("Herramienta de Análisis de Llamadas")

# Crear el menú en la barra lateral
menu = st.sidebar.radio("Menú de Navegación", ("Home", "Analizador de llamadas", "Chatbot"))

# Entrada de la API Key (se mantiene accesible en todas las secciones que la requieran)
st.sidebar.header("Configuración")
api_key_input = st.sidebar.text_input("Introduce tu OpenAI API Key", type="password")
st.session_state['api_key'] = api_key_input  # Guardar la API Key en el estado

# Agregar estilos CSS personalizados
st.markdown(
    """
    <style>
    /* Contenedor principal */
    .main-container {
        display: flex;
        flex-direction: column;
        height: calc(100vh - 150px); /* Ajustar según sea necesario */
        margin: 0;
        padding: 0;
    }
    /* Historial de chat */
    .chat-history {
        flex: 1;
        overflow-y: auto;
        padding: 10px;
    }
    /* Barra de entrada de chat fija */
    .chat-input {
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: white;
        padding: 10px;
        box-shadow: 0 -2px 5px rgba(0,0,0,0.1);
    }
    /* Mensajes del usuario */
    .user-message {
        background-color: #009dac;
        color: white;
        padding: 10px;
        border-radius: 10px;
        max-width: 70%;
        margin-bottom: 10px;
        align-self: flex-end;
        margin-left: auto;
    }
    /* Mensajes del asistente */
    .assistant-message {
        background-color: #F1F0F0;
        color: black;
        padding: 10px;
        border-radius: 10px;
        max-width: 70%;
        margin-bottom: 10px;
        align-self: flex-start;
    }
    /* Título del chat */
    .chat-title {
        margin-bottom: 10px;
    }
    /* Reducir espacio entre título y mensajes */
    .streamlit-expanderHeader {
        margin-bottom: 0px;
    }
    /* Ajustar el margen del título principal */
    .block-container > .main > div > div > div > div > div > h1 {
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Contenido según la selección del menú
if menu == "Home":
    st.markdown("""
    ## Bienvenido a la Herramienta de Análisis de Llamadas

    Esta herramienta te permite analizar y transcribir llamadas de servicio al cliente para obtener información valiosa sobre el tipo de llamadas, las razones, la información solicitada, la resolución, el sentimiento y observaciones adicionales.

    ### **Características:**
    - **Transcripción de Llamadas:** Convierte audio a texto utilizando Whisper.
    - **Diarización de Parlantes:** Identifica diferentes hablantes en la llamada.
    - **Análisis con GPT-4:** Clasifica y analiza cada llamada en categorías específicas.
    - **Chatbot de Soporte:** Interactúa con un asistente para obtener respuestas sobre los análisis realizados.
    - **Generación de Reportes:** Descarga los análisis en formato Excel o JSON.

    ### **Cómo Usar la Herramienta:**
    1. **Analizador de Llamadas:**
       - Sube un archivo de audio (individual o en ZIP) para comenzar el análisis.
       - Espera a que se complete la transcripción y el análisis.
       - Descarga los resultados en los formatos disponibles.

    2. **Chatbot:**
       - Realiza preguntas sobre los análisis de llamadas para obtener respuestas rápidas y detalladas.

    ### **Requisitos:**
    - Una clave válida de la API de OpenAI.
    """)

elif menu == "Analizador de llamadas":
    if st.session_state['api_key']:
        # Opción para seleccionar entre análisis de una llamada o varias
        analysis_type = st.radio("Selecciona tipo de análisis", ("Análisis de una llamada", "Análisis de varias llamadas (ZIP)"))

        if analysis_type == "Análisis de una llamada":
            uploaded_file = st.file_uploader("Sube un archivo de audio", type=["mp3", "wav", "m4a", "flac", "ogg"])

            if uploaded_file:
                if uploaded_file.name not in st.session_state['processed_files']:
                    st.audio(uploaded_file, format="audio/mp3")
                    st.write("Transcribiendo y analizando el audio... Por favor espera.")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_audio:
                        temp_audio.write(uploaded_file.read())
                        temp_audio_path = temp_audio.name

                    analysis = analyze_single_call(temp_audio_path, st.session_state['api_key'])

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
                analysis_results = analyze_multiple_calls(uploaded_zip, st.session_state['api_key'])

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
        st.warning("Por favor, introduce tu OpenAI API Key en la sección de Configuración.")

elif menu == "Chatbot":
    if st.session_state['api_key']:
        st.header("Chat de Soporte")

        # Contenedor principal
        chat_container = st.container()
        with chat_container:
            st.markdown('<div class="main-container">', unsafe_allow_html=True)

            # Historial de Chat
            st.markdown('<div class="chat-history">', unsafe_allow_html=True)
            if st.session_state['chat_history']:
                for chat in st.session_state['chat_history']:
                    if chat['usuario']:
                        # Mensaje del Usuario
                        st.markdown(f"""
                        <div class="user-message">
                            {chat['usuario']}
                        </div>
                        """, unsafe_allow_html=True)
                    if chat['asistente']:
                        # Mensaje del Asistente
                        st.markdown(f"""
                        <div class="assistant-message">
                            {chat['asistente']}
                        </div>
                        """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Barra de Entrada de Chat
            st.markdown('<div class="chat-input">', unsafe_allow_html=True)
            col1, col2 = st.columns([4, 1])
            with col1:
                user_input = st.text_input(
                    "Escribe tu pregunta sobre los análisis de las llamadas:",
                    key="user_message"
                )
            with col2:
                send_button = st.button("Enviar")
            
            # Manejar el envío del mensaje
            if send_button and user_input:
                if st.session_state['analysis_results']:
                    chat_response = get_chat_response(user_input, st.session_state['analysis_results'], st.session_state['api_key'])
                    st.session_state['chat_history'].append({"usuario": user_input, "asistente": chat_response})
                    st.session_state['user_message'] = ''  # Clear the input after processing
                else:
                    st.warning("Por favor, realiza primero el análisis de las llamadas.")

            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)  # Cerrar main-container
    else:
        st.warning("Por favor, introduce tu OpenAI API Key en la sección de Configuración.")


