import tkinter as tk
from tkinter import ttk, scrolledtext

def create_chat_tab(app, parent):
    """Create the chat interface in the given parent widget"""
    # Create ChatbotUI instance with app's AI response generator
    chatbot = ChatbotUI(lambda msg: app._generate_ai_response(msg))
    # Create the actual chat interface
    chatbot.create_chat_tab(parent)
    # Store chatbot instance on app for later reference
    app.chatbot = chatbot

def open_prompt_tuner(app):
    """Open the prompt tuning window"""
    if hasattr(app, 'chatbot'):
        app.chatbot.open_prompt_tuner()

class ChatbotUI:
    def __init__(self, send_callback):
        """
        send_callback: a function that receives a message string and returns a response string.
        """
        self.send_callback = send_callback
        self.chat_input = None
        self.txt_chat_log = None

    def create_chat_tab(self, parent):
        """Create chat interface tab"""
        # Chat history display
        self.txt_chat_log = scrolledtext.ScrolledText(
            parent, 
            wrap=tk.WORD,
            bg="black",
            fg="green",
            insertbackground="green"
        )
        self.txt_chat_log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.txt_chat_log.insert(tk.END, "AI Chat Interface Ready...\n")
        self.txt_chat_log.config(state=tk.DISABLED)

        # Chat input area
        input_frame = ttk.Frame(parent)
        input_frame.pack(fill=tk.X, padx=5, pady=5)

        self.chat_input = tk.StringVar()
        entry = ttk.Entry(
            input_frame, 
            textvariable=self.chat_input
        )
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        entry.bind("<Return>", lambda e: self.chat_send_message())

        send_btn = ttk.Button(
            input_frame,
            text="Send",
            command=self.chat_send_message
        )
        send_btn.pack(side=tk.RIGHT, padx=(5, 0))

    def chat_send_message(self):
        message = self.chat_input.get().strip()
        if not message:
            return
        self.chat_input.set("")
        self.append_chat("You", message)
        response = self.send_callback(message)
        self.append_chat("AI", response)

    def append_chat(self, sender, message):
        if self.txt_chat_log:
            self.txt_chat_log.config(state=tk.NORMAL)
            self.txt_chat_log.insert(tk.END, f"{sender}: {message}\n")
            self.txt_chat_log.see(tk.END)
            self.txt_chat_log.config(state=tk.DISABLED)

    def open_prompt_tuner(self):
        """Open prompt tuning window"""
        tuner = tk.Toplevel(self)
        tuner.title("AI Prompt Tuner")
        tuner.geometry("600x400")
        tuner.configure(bg="black")

        # Add prompt tuning interface
        ttk.Label(
            tuner,
            text="Adjust AI Prompt Parameters",
            style="Chat.TLabel"
        ).pack(pady=10)

        # Add prompt parameters
        parameters = {
            "Temperature": (0.1, 2.0, 0.7),
            "Top P": (0.1, 1.0, 0.9),
            "Response Length": (50, 1000, 250),
        }

        for param, (min_val, max_val, default) in parameters.items():
            frame = ttk.Frame(tuner)
            frame.pack(fill=tk.X, padx=10, pady=5)
            
            ttk.Label(
                frame, 
                text=param,
                style="Chat.TLabel"
            ).pack(side=tk.LEFT)

            scale = ttk.Scale(
                frame,
                from_=min_val,
                to=max_val,
                value=default,
                orient=tk.HORIZONTAL
            )
            scale.pack(side=tk.RIGHT, fill=tk.X, expand=True)
