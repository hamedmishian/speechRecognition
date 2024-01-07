from vosk import Model, KaldiRecognizer
from happytransformer import HappyGeneration, GENSettings
import pyaudio
import time

# Vosk setup
model = Model(r"C:\Users\lenovo\PycharmProjects\Vosk\vosk-model-small-en-us-0.15")
recognizer = KaldiRecognizer(model, 16000)

# Audio setup
mic = pyaudio.PyAudio()
stream = mic.open(format=pyaudio.paInt16, channels=1,
                  rate=16000, input=True, frames_per_buffer=8192)
stream.start_stream()

# Text file for user inputs (accumulate over time)
user_input_file_path = r"C:\Users\lenovo\PycharmProjects\Vosk\user_inputs.txt"
recognized_texts = []

try:
    start_time = time.time()

    while (time.time() - start_time) <= 5:
        data = stream.read(4096, exception_on_overflow=False)

        if recognizer.AcceptWaveform(data):
            text = recognizer.Result()
            recognized_text = text[14:-3].strip()
            print("User: ", recognized_text)

            recognized_texts.append(recognized_text)

    # Save accumulated user inputs to the text file
    with open(user_input_file_path, 'w') as user_input_file:
        user_input_file.write("\n".join(recognized_texts))

    # Combine all recognized texts as a question for text generation
    combined_text = " ".join(recognized_texts)

    happy_gen = HappyGeneration("GPT-2", "gpt2")
    happy_gen.train("schumann.txt")
    args = GENSettings(max_length=100)
    response = happy_gen.generate_text(
        combined_text, args=args)

    print("AI Response (GPT-2): ", response.text)

except KeyboardInterrupt:
    print("Recording stopped.")

stream.stop_stream()
stream.close()
mic.terminate()
