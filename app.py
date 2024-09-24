import streamlit as st
import librosa
import whisper
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import webrtcvad
import requests
import tempfile

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

# Whisper transcription function
def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result

# Function to load the labeled transcription from file
def load_transcription(file_path):
    with open(file_path, 'r') as file:
        transcription = file.readlines()
    return transcription

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

# Streamlit interface
st.title("Call Analysis Tool")

# Sidebar for API key input
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

# Upload an audio file
uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])

if uploaded_file is not None and api_key:
    st.audio(uploaded_file, format="audio/mp3")

    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
        temp_audio_file.write(uploaded_file.read())
        temp_audio_path = temp_audio_file.name

    # Load and process audio
    audio, sr = librosa.load(temp_audio_path, sr=None)

    # Diarize the audio
    speaker_labels, speech_indices = diarize_audio(audio, sr)

    # Transcribe the audio using Whisper
    st.write("Transcribing the audio... Please wait.")
    result = transcribe_audio(temp_audio_path)

    # Align transcription with speaker diarization
    transcript_with_speakers = []
    for segment in result['segments']:
        word_start = segment['start']
        nearest_index = np.argmin(np.abs(speech_indices - (word_start * sr)))
        speaker = speaker_labels[nearest_index]
        transcript_with_speakers.append(f"Speaker {speaker}: {segment['text']}")

    # Save the transcript with speaker labels to a file
    with open('labeled_transcript.txt', 'w') as f:
        for line in transcript_with_speakers:
            f.write(line + '\n')

    # Display the transcription with speaker labels
    st.write("\nTranscription with Speaker Labels:")
    for line in transcript_with_speakers:
        st.write(line)

    # Load the labeled transcription for analysis
    transcription = load_transcription('labeled_transcript.txt')

    # Create the prompt for GPT
    prompt = create_prompt(transcription)

    # Analyze the call using GPT
    st.write("Analyzing the call... Please wait.")
    analysis = analyze_call_with_gpt_mini(prompt, api_key)

    # Display the analysis result
    st.write("\nAnalysis Result:")
    st.write(analysis)

    # Optionally, download the labeled transcript
    transcript_text = "\n".join(transcript_with_speakers)
    st.download_button(label="Download Labeled Transcript", data=transcript_text, file_name="labeled_transcript.txt")
