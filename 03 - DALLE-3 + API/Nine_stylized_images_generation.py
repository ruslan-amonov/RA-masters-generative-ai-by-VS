import os
from openai import OpenAI

class ImageGenerator:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.styles = [
            "funny",
            "formal",
            "cartoon",
            "realistic",
            "black and white",
            "old picture",
            "futuristic",
            "ancient times",
            "casual"
        ]

    def generate_image(self, prompt: str, style: str) -> str:
        try:
            response = self.client.images.generate(
                prompt=f"{prompt} in {style} style",
                n=1,
                size="1024x1024",
                model="dall-e-3"
            )
            return response.data[0].url
        except Exception as e:
            print(f"Error generating {style} image: {e}")
            return None

    def generate_images(self, prompt: str) -> dict:
        images = {}
        for style in self.styles:
            url = self.generate_image(prompt, style)
            if url:
                images[style] = url
        return images

    def save_images(self, images: dict, output_file: str):
        with open(output_file, 'w') as f:
            for style, url in images.items():
                f.write(f"{style}: {url}\n")


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = input("Please enter your OpenAI API key: ")

    prompt = input("Please enter the prompt for image generation: ")
    output_file = input("Please enter the output file name (with .txt extension): ")

    generator = ImageGenerator(api_key)

    images = generator.generate_images(prompt)

    for style, url in images.items():
        print(f"Generated {style} image: {url}")

    generator.save_images(images, output_file)
    print("All images generated and saved successfully!")


if __name__ == "__main__":
    main()