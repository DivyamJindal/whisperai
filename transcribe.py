import sys
import whisper
import torch
import ssl
import ffmpeg
import os
from tempfile import NamedTemporaryFile

# Create unverified SSL context to handle certificate issues
ssl._create_default_https_context = ssl._create_unverified_context

def extract_audio(video_path):
    try:
        # Create a temporary file for the extracted audio
        with NamedTemporaryFile(suffix='.mp3', delete=False) as temp_audio:
            temp_audio_path = temp_audio.name

        # Extract audio from video using ffmpeg
        print("Extracting audio from video...")
        # Properly escape the file paths for ffmpeg
        stream = ffmpeg.input(video_path)
        stream = ffmpeg.output(stream, temp_audio_path, acodec='libmp3lame')
        ffmpeg.run(stream, capture_stdout=True, capture_stderr=True, overwrite_output=True)

        return temp_audio_path
    except Exception as e:
        print(f"Error extracting audio: {str(e)}")
        return None

def transcribe_audio(file_path):
    try:
        # Check if the file is a video (mp4)
        is_video = file_path.lower().endswith('.mp4')
        audio_path = extract_audio(file_path) if is_video else file_path

        if is_video and not audio_path:
            return False

        # Load the Whisper model (using base model for faster processing)
        print("Loading Whisper model...")
        model = whisper.load_model("base")

        # Transcribe the audio
        print(f"Transcribing audio...")
        result = model.transcribe(audio_path)

        # Clean up temporary audio file if it was created
        if is_video and os.path.exists(audio_path):
            os.unlink(audio_path)

        # Print the transcription
        print("\nTranscription:")
        print(result["text"])
        return True

    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def main():
    if len(sys.argv) != 2:
        print("Usage: python transcribe.py <path_to_audio_file>")
        return

    audio_path = sys.argv[1]
    transcribe_audio(audio_path)

if __name__ == "__main__":
    main()