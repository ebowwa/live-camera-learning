
import pathlib
import textwrap
import google.generativeai as genai
import PIL.Image
import os

# IMPORTANT: Set your GOOGLE_API_KEY as an environment variable before running this script.
# For example, in your terminal:
# export GOOGLE_API_KEY="YOUR_API_KEY"

# Fetch the API key from an environment variable
GOOGLE_API_KEY = os.getenv('GEMINI_API_KEY')

if not GOOGLE_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set. Please set it before running the script.")

genai.configure(api_key=GOOGLE_API_KEY)

# Create the model
model = genai.GenerativeModel('gemini-2.5-flash')

# Load an image
try:
    img = PIL.Image.open('snapshots/snapshot_20250804_204004.png')
except FileNotFoundError:
    print("Error: Image file not found. Make sure 'snapshots/snapshot_20250804_204004.png' exists.")
    exit()

# Ask a question about the image
try:
    response = model.generate_content(["Describe one word of the object in this picture", img])
    print(response.text)
except Exception as e:
    print(f"An error occurred while calling the Gemini API: {e}")

