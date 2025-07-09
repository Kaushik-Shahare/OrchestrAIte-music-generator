from transformers import pipeline
import soundfile as sf

# Initialize the Bark TTS pipeline
synth = pipeline("text-to-speech", "suno/bark")

# The text you want to synthesize
text = "This is a test of Bark text to speech running on a Mac M1."

# Generate the audio
tts_out = synth(text)

# Extract audio and sample rate
audio = tts_out["audio"]
sr = tts_out["sampling_rate"]

# Save to a WAV file
sf.write("bark_test.wav", audio, sr)
print("Audio saved as bark_test.wav")