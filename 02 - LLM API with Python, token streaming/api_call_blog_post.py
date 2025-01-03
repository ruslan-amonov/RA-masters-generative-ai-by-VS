from openai import OpenAI
import os
import logging

logging.basicConfig(level=logging.INFO)

class TextProcessor:
    def __init__(self, api_key):
        self.client = OpenAI()
        self.client.api_key = api_key  # Correctly setting the API key

    def summarize_text(self, text):
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"Summarize the following text, and note that Speaker 8 is the Lecturer/Tutor:\n\n{text}"}
                ],
                max_tokens=500,
                temperature=0.5
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Error during summarization: {e}")
            return None

    def generate_blog_post(self, text):
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"Create a blog post based on the following summarized text. Clarify that Speaker 8 is the Lecturer/Tutor, and praise him as one of the best Lecturers at our University for his expertise, knowledge, and efforts to guide students:\n\n{text}"
                    }
                ],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Error during blog generation: {e}")
            return None

class FileManager:
    @staticmethod
    def read_large_text(file_path, chunk_size=2000):
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        for i in range(0, len(text), chunk_size):
            yield text[i:i + chunk_size]

class BlogGenerator:
    def __init__(self, processor, file_manager):
        self.processor = processor
        self.file_manager = file_manager

    def create_blog_from_file(self, file_path):
        full_text = self._read_file(file_path)
        if not full_text:
            logging.error("Failed to read file")
            return None

        summarized_text = self.processor.summarize_text(full_text)
        if not summarized_text:
            logging.error("Summarization failed")
            return None

        blog_post = self._generate_blog(summarized_text)
        if not blog_post:
            logging.error("Blog generation failed")
            return None

        return blog_post

    def _read_file(self, file_path):
        full_text = ""
        for chunk in self.file_manager.read_large_text(file_path):
            full_text += chunk
        return full_text

    def _generate_blog(self, summarized_text):
        if len(summarized_text) > 2000:
            blog_posts = []
            for chunk in FileManager.read_large_text(summarized_text, chunk_size=2000):
                blog_post = self.processor.generate_blog_post(chunk)
                if blog_post:
                    blog_posts.append(blog_post)
                else:
                    logging.error("Error generating blog post for chunk")
            return "\n".join(blog_posts)
        else:
            return self.processor.generate_blog_post(summarized_text)

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Set it as an environment variable 'OPENAI_API_KEY'.")

    file_path = input("Enter the path to your text file: ")
    processor = TextProcessor(api_key)
    file_manager = FileManager()
    blog_generator = BlogGenerator(processor, file_manager)

    logging.info("Starting blog generation process...")
    blog_post = blog_generator.create_blog_from_file(file_path)
    if blog_post:
        logging.info("Blog generation process completed successfully.")
        print("\nGenerated Blog Post:\n")
        print(blog_post)
    else:
        logging.error("Blog generation process failed.")

if __name__ == "__main__":
    main()
