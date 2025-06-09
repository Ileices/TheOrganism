import tkinter as tk
from tkinter import ttk, scrolledtext

class ChatbotUI:
    def __init__(self, send_callback):
        """
        send_callback: a function that receives a message string and returns a response string.
        """
        self.send_callback = send_callback
        self.chat_input = None
        self.txt_chat_log = None

    def create_chat_tab(self, parent):
        # Create chat tab UI components
        label = ttk.Label(parent, text="Chat with the AI (Demo/Placeholder):")
        label.pack(anchor="w", padx=5, pady=5)
        self.txt_chat_log = scrolledtext.ScrolledText(parent, wrap=tk.WORD, height=10, bg="black", fg="green", insertbackground="green")
        self.txt_chat_log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        entry_frame = ttk.Frame(parent)
        entry_frame.pack(fill=tk.X, padx=5, pady=5)
        self.chat_input = tk.StringVar()
        chat_entry = ttk.Entry(entry_frame, textvariable=self.chat_input, width=50)
        chat_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        chat_entry.bind("<Return>", lambda event: self.chat_send_message())
        send_btn = ttk.Button(entry_frame, text="Send", command=self.chat_send_message)
        send_btn.pack(side=tk.LEFT, padx=5)
        # Add Prompt Tuner Button
        tuner_btn = ttk.Button(parent, text="Prompt Tuner", command=self.open_prompt_tuner)
        tuner_btn.pack(pady=5)

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
        tuner_win = tk.Toplevel()
        tuner_win.title("Prompt Tuner")
        tuner_win.geometry("600x400")
        tuner_win.configure(bg="black")
        prompt_label = tk.Label(tuner_win, text="Enter your project goal:", bg="black", fg="green")
        prompt_label.pack(pady=5)
        prompt_entry = tk.Entry(tuner_win, width=50, bg="black", fg="green")
        prompt_entry.pack(pady=5)
        tuner_result = scrolledtext.ScrolledText(tuner_win, wrap=tk.WORD, bg="black", fg="green", height=10)
        tuner_result.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        def refine_prompt():
            goal = prompt_entry.get()
            refined = f'{{"goal": "{goal}", "instructions": "Refined JSON prompt based on the provided goal."}}'
            tuner_result.delete("1.0", tk.END)
            tuner_result.insert(tk.END, refined)
        ttk.Button(tuner_win, text="Tune Prompt", command=refine_prompt).pack(pady=5)
