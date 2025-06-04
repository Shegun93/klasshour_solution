import fitz
import os
from tqdm import tqdm

pdf_folder = "/home/shegun93/Klasshour_Rags/Phy"
output_folder = "/home/shegun93/Klasshour_Rags/Phy/Extracted_Text"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
# Dictionary to hold the extracted text
text = {}

for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, filename)
        doc = fitz.open(pdf_path)
        text[filename] = ""
        
        for page in tqdm(doc):
            text[filename] += page.get_text()
        
        doc.close()
        # Save the extracted text to a file
        output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text[filename])
        print(f"Extracted text from {filename} and saved to {output_path}")
# Print the number of files processed
print(f"Processed {len(text)} PDF files.")