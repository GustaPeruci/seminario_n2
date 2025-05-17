import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from tkinter import messagebox
from threading import Thread

from model.topic_modeler import TopicModeler

class TopicApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Modelagem de Tópicos em Jurisprudência")
        self.root.geometry("900x600")

        self._build_ui()

    def _build_ui(self):
        # Frame de seleção de modelo
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill='x')

        ttk.Label(control_frame, text="Escolha o Modelo:").pack(side="left", padx=5)

        self.model_var = tk.StringVar(value="nmf")
        ttk.Radiobutton(control_frame, text="NMF", variable=self.model_var, value="nmf").pack(side="left")
        ttk.Radiobutton(control_frame, text="LDA", variable=self.model_var, value="lda").pack(side="left")

        ttk.Label(control_frame, text="Número de Tópicos:").pack(side="left", padx=10)
        self.n_topics_spin = ttk.Spinbox(control_frame, from_=2, to=20, width=5)
        self.n_topics_spin.set(5)
        self.n_topics_spin.pack(side="left")

        self.run_button = ttk.Button(control_frame, text="Executar Modelagem", command=self.run_model)
        self.run_button.pack(side="right", padx=10)

        # Loading label
        self.loading_label = ttk.Label(self.root, text="", foreground="blue")
        self.loading_label.pack(pady=10)

        # Text area de resultados
        self.result_area = ScrolledText(self.root, wrap="word", font=("Courier", 10))
        self.result_area.pack(fill='both', expand=True, padx=10, pady=5)

    def run_model(self):
        self.run_button.config(state='disabled')
        self.loading_label.config(text="Processando... aguarde (pode demorar alguns segundos)...")
        self.result_area.delete("1.0", tk.END)

        def thread_task():
            try:
                modeler = TopicModeler()
                n_topics = int(self.n_topics_spin.get())
                topics, _ = modeler.run(model_type=self.model_var.get(), n_topics=n_topics)

                self._display_topics(topics)
            except Exception as e:
                messagebox.showerror("Erro", str(e))
            finally:
                self.loading_label.config(text="")
                self.run_button.config(state='normal')

        Thread(target=thread_task).start()

    def _display_topics(self, topics):
        self.result_area.insert(tk.END, "📚 Tópicos Extraídos:\n\n")
        for idx, words in topics.items():
            title = self._generate_topic_title(words)
            self.result_area.insert(tk.END, f"🔹 Tópico {idx+1} - {title}\n")
            self.result_area.insert(tk.END, "   " + ", ".join(words) + "\n\n")

    def _generate_topic_title(self, words):
        # Usa as 2 palavras mais fortes como título resumido
        return " / ".join(words[:2]).capitalize()

if __name__ == "__main__":
    root = tk.Tk()
    app = TopicApp(root)
    root.mainloop()
