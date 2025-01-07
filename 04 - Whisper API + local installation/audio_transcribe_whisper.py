import whisper


class WhisperTranscriber:
    def __init__(self, model_name: str):
        self.model = self.load_model(model_name)

    def load_model(self, model_name: str):
        return whisper.load_model(model_name)

    def transcribe_audio(self, input_file_path: str) -> str:
        result = self.model.transcribe(input_file_path)
        return result["text"]

    def save_transcription(self, transcription: str, output_file_name: str):
        with open(output_file_name, 'w') as f:
            f.write(transcription)


def main():
    model_name = input("Please enter the Whisper model to use (e.g., 'base', 'small', 'medium', 'large'): ")
    input_file_path = input("Please enter the audio file path: ")
    output_file_name = input("Please enter the output file name (with .txt extension): ")

    transcriber = WhisperTranscriber(model_name)

    transcription = transcriber.transcribe_audio(input_file_path)
    print("Transcription: ", transcription)
    transcriber.save_transcription(transcription, output_file_name)
    print(f"Transcription saved to {output_file_name}")


if __name__ == "__main__":
    main()
