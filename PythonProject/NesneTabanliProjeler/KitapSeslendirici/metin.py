import os
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
from gtts import gTTS
import PyPDF2

class PDFToSpeechApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Book App")
        self.root.geometry("350x300")
        self.root.configure(bg="#f0f0f0")

        self.languages = {
            "Turkish": "tr",
            "English": "en",
            "French": "fr",
            "German": "de",
            "Spanish": "es",
            "Italian": "it",
        }

        self.selected_language = tk.StringVar(value="Turkish")

        self.setup_ui()

    def setup_ui(self):
        # Başlık
        title = tk.Label(self.root, text="Audio Book App", font=("Arial", 16, "bold"), bg="#f0f0f0")
        title.pack(pady=10)

        # Dil seçimi
        lang_label = tk.Label(self.root, text="Select Language:", bg="#f0f0f0")
        lang_label.pack()

        self.lang_combobox = ttk.Combobox(
            self.root,
            textvariable=self.selected_language,
            values=list(self.languages.keys()),
            state="readonly",
            width=20,
            font=("Arial", 10)
        )
        self.lang_combobox.pack(pady=5)

        # PDF Seçme Butonu
        select_btn = tk.Button(
            self.root, text="Choose PDF", command=self.select_file,
            padx=20, pady=10, bg="#4CAF50", fg="white", relief="raised", font=("Arial", 10, "bold")
        )
        select_btn.pack(pady=15)

        # Çıkış butonu ve görsel
        exit_img = Image.open("exit.png").resize((30, 30), Image.LANCZOS)
        self.exit_icon = ImageTk.PhotoImage(exit_img)

        exit_btn = tk.Button(
            self.root, image=self.exit_icon, command=self.root.quit,
            bg="#f0f0f0", borderwidth=0, activebackground="#e0e0e0"
        )
        exit_btn.place(x=310, y=260)

    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if file_path:
            text = self.extract_text_from_pdf(file_path)
            language_code = self.languages.get(self.selected_language.get(), "tr")
            self.convert_text_to_speech(text, "output.mp3", language_code)
            print("Audio file saved as output.mp3")

    def extract_text_from_pdf(self, path):
        text = ""
        with open(path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
        return text

    def convert_text_to_speech(self, text, output_path, language):
        if not text.strip():
            print("No text found in PDF.")
            return
        tts = gTTS(text=text, lang=language)
        tts.save(output_path)

# Uygulama başlatılıyor
if __name__ == "__main__":
    root = tk.Tk()
    app = PDFToSpeechApp(root)
    root.mainloop()
