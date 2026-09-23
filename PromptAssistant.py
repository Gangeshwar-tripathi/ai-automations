import json
import threading
import urllib.request
import tkinter as tk
from tkinter import scrolledtext

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "llama3.2:3b"


def ask_ai(prompt):

    payload = {
        "model": MODEL,
        "prompt": f"""
You are a helpful technical AI assistant.

Analyze the user's request and provide a practical possible solution.

Give:
1. Problem understanding
2. Possible cause
3. Recommended solution
4. Commands/code if useful

Be concise and actionable.

User request:
{prompt}
""",
        "stream": False
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(request, timeout=300) as response:
        result = json.loads(response.read().decode())

    return result["response"]


class App:

    def __init__(self):

        self.root = tk.Tk()

        self.root.title("🤖 Private AI Assistant")

        self.root.geometry("700x600")

        self.root.attributes("-topmost", True)

        self.root.configure(bg="#151515")

        title = tk.Label(
            self.root,
            text="🤖 PRIVATE AI ASSISTANT",
            font=("Arial", 22, "bold"),
            fg="white",
            bg="#151515"
        )

        title.pack(pady=(20, 10))

        tk.Label(
            self.root,
            text="Type your problem or question",
            font=("Arial", 13),
            fg="#bbbbbb",
            bg="#151515"
        ).pack()

        self.prompt = scrolledtext.ScrolledText(
            self.root,
            height=6,
            wrap=tk.WORD,
            font=("Arial", 14),
            bg="#242424",
            fg="white",
            insertbackground="white"
        )

        self.prompt.pack(
            fill=tk.X,
            padx=20,
            pady=15
        )

        self.prompt.insert(
            "1.0",
            "Explain this Python error and give me a possible solution."
        )

        self.button = tk.Button(
            self.root,
            text="🤖 Ask AI",
            font=("Arial", 15, "bold"),
            command=self.ask,
            padx=25,
            pady=10
        )

        self.button.pack(pady=5)

        self.status = tk.Label(
            self.root,
            text="Ready",
            font=("Arial", 12),
            fg="#55dd88",
            bg="#151515"
        )

        self.status.pack(pady=5)

        tk.Label(
            self.root,
            text="Possible solution",
            font=("Arial", 15, "bold"),
            fg="white",
            bg="#151515"
        ).pack(anchor="w", padx=20, pady=(10, 5))

        self.answer = scrolledtext.ScrolledText(
            self.root,
            wrap=tk.WORD,
            font=("Arial", 14),
            bg="#202020",
            fg="white",
            insertbackground="white"
        )

        self.answer.pack(
            fill=tk.BOTH,
            expand=True,
            padx=20,
            pady=(0, 20)
        )

    def ask(self):

        prompt = self.prompt.get(
            "1.0",
            tk.END
        ).strip()

        if not prompt:
            return

        self.button.config(
            state=tk.DISABLED
        )

        self.status.config(
            text="🤔 Thinking..."
        )

        self.answer.delete(
            "1.0",
            tk.END
        )

        self.answer.insert(
            tk.END,
            "Generating solution..."
        )

        threading.Thread(
            target=self.run_ai,
            args=(prompt,),
            daemon=True
        ).start()

    def run_ai(self, prompt):

        try:

            result = ask_ai(prompt)

            self.root.after(
                0,
                lambda: self.show_result(result)
            )

        except Exception as error:

            self.root.after(
                0,
                lambda: self.show_result(
                    "Error connecting to Ollama:\n\n"
                    + str(error)
                    + "\n\nMake sure Ollama is running."
                )
            )

    def show_result(self, result):

        self.answer.delete(
            "1.0",
            tk.END
        )

        self.answer.insert(
            tk.END,
            result
        )

        self.status.config(
            text="✓ Solution ready"
        )

        self.button.config(
            state=tk.NORMAL
        )


app = App()

app.root.mainloop()
