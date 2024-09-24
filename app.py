import streamlit as st
import librosa
import whisper
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import webrtcvad

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
    clustering = AgglomerativeClustering(n_clusters=num_speakers, affinity='cosine', linkage='average')
    speaker_labels = clustering.fit_predict(mfccs)

    return speaker_labels, speech_indices

# Whisper transcription function
def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result

# Streamlit interface
st.title("Call Analysis Tool")

# Upload an audio file
uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/mp3")

    # Load and process audio
    audio, sr = librosa.load(uploaded_file, sr=None)

    # Diarize the audio
    speaker_labels, speech_indices = diarize_audio(audio, sr)

    # Transcribe the audio
    st.write("Transcribing the audio... Please wait.")
    result = transcribe_audio(uploaded_file)

    # Align transcription with speaker diarization
    transcript_with_speakers = []
    for segment in result['segments']:
        word_start = segment['start']
        nearest_index = np.argmin(np.abs(speech_indices - (word_start * sr)))
        speaker = speaker_labels[nearest_index]
        transcript_with_speakers.append(f"Speaker {speaker}: {segment['text']}")

    # Display the transcription with speaker labels
    st.write("\nTranscription with Speaker Labels:")
    for line in transcript_with_speakers:
        st.write(line)

    # Generate a labeled transcript .txt file
    transcript_text = "\n".join(transcript_with_speakers)
    st.download_button(label="Download Labeled Transcript", data=transcript_text, file_name="labeled_transcript.txt")
