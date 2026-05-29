from supertonic import TTS

# First run downloads the model from Hugging Face automatically.
tts = TTS(auto_download=True)

style = tts.get_voice_style(voice_name="M1")

text = "Why did my $6202.39 vanish?! <sigh> $NVDA & AMZN crashed 15.44% today... C'est fini! <laugh> #RIP_my_wallets!"

wav, duration = tts.synthesize(
    text=text,
    lang="en",                      # Language code (e.g., "en", "ko", "na" for language-agnostic)
    voice_style=style,              # Voice style object
    total_steps=8,                  # Quality: 5 (low) to 12 (high), default 8 (medium)
    speed=1.05,                     # Speed: 0.7 (slow) to 2.0 (fast)
)
# wav: numpy array of shape (1, num_samples,) with dtype=np.float32, sampled at 44100 Hz
# duration: numpy array of shape (1,) containing the duration of the generated audio in seconds

tts.save_audio(wav, "output.wav")
# import soundfile as sf
# sf.write("output.wav", wav.squeeze(), 44100)

print(f"Generated {duration[0]:.2f}s of audio")