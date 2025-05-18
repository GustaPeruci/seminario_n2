
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
from threading import Thread
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from model.topic_modeler import TopicModeler

class TopicModelingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Análise de Tópicos em Jurisprudência")
        self.root.geometry("1200x800")
        
        self.setup_styles()
        self.create_widgets()
        
    def setup_styles(self):
        style = ttk.Style()
        style.configure("Title.TLabel", font=("Helvetica", 16, "bold"))
        style.configure("Subtitle.TLabel", font=("Helvetica", 12))
        
    def create_widgets(self):
        # Header
        header = ttk.Frame(self.root, padding="20 10")
        header.pack(fill='x')
        
        ttk.Label(header, text="Modelagem de Tópicos em Jurisprudência", 
                 style="Title.TLabel").pack()
        
        # Control Panel
        controls = ttk.LabelFrame(self.root, text="Configurações", padding="10")
        controls.pack(fill='x', padx=10, pady=5)
        
        # Model Selection
        model_frame = ttk.Frame(controls)
        model_frame.pack(fill='x', pady=5)
        
        ttk.Label(model_frame, text="Modelo:").pack(side="left", padx=5)
        self.model_var = tk.StringVar(value="nmf")
        ttk.Radiobutton(model_frame, text="NMF", variable=self.model_var, 
                       value="nmf").pack(side="left", padx=10)
        ttk.Radiobutton(model_frame, text="LDA", variable=self.model_var, 
                       value="lda").pack(side="left", padx=10)
        
        # Topics Control
        topics_frame = ttk.Frame(controls)
        topics_frame.pack(fill='x', pady=5)
        
        ttk.Label(topics_frame, text="Número de Tópicos:").pack(side="left", padx=5)
        self.n_topics_spin = ttk.Spinbox(topics_frame, from_=2, to=20, width=5)
        self.n_topics_spin.set(5)
        self.n_topics_spin.pack(side="left", padx=5)
        
        self.run_button = ttk.Button(topics_frame, text="Executar Análise", 
                                   command=self.run_analysis)
        self.run_button.pack(side="right", padx=10)
        
        # Results Area
        results_frame = ttk.Frame(self.root)
        results_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Topics Display
        topics_frame = ttk.LabelFrame(results_frame, text="Tópicos Identificados")
        topics_frame.pack(side='left', fill='both', expand=True, padx=5)
        
        self.topics_text = ScrolledText(topics_frame, wrap="word", 
                                      font=("Courier", 10), width=50)
        self.topics_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Visualization Frame
        viz_frame = ttk.LabelFrame(results_frame, text="Visualização")
        viz_frame.pack(side='right', fill='both', expand=True, padx=5)
        
        self.canvas_frame = ttk.Frame(viz_frame)
        self.canvas_frame.pack(fill='both', expand=True)
        
        # Status
        self.status_label = ttk.Label(self.root, text="", foreground="blue")
        self.status_label.pack(pady=5)
        
    def run_analysis(self):
        self.run_button.config(state='disabled')
        self.status_label.config(text="Processando análise... Por favor, aguarde.")
        self.topics_text.delete("1.0", tk.END)
        
        def analysis_task():
            try:
                modeler = TopicModeler()
                n_topics = int(self.n_topics_spin.get())
                topics, fig = modeler.run(model_type=self.model_var.get(), 
                                        n_topics=n_topics)
                
                self.root.after(0, lambda: self.display_results(topics, fig))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Erro", str(e)))
            finally:
                self.root.after(0, lambda: self.cleanup_analysis())
        
        Thread(target=analysis_task).start()
    
    def display_results(self, topics, fig):
        self.topics_text.delete("1.0", tk.END)
        self.topics_text.insert(tk.END, "📊 Resultados da Análise\n\n")
        
        for idx, topic_info in topics.items():
            self.topics_text.insert(tk.END, f"🔹 Tópico {idx+1}\n")
            self.topics_text.insert(tk.END, f"Título: {topic_info['title']}\n")
            self.topics_text.insert(tk.END, f"Palavras-chave: {', '.join(topic_info['words'])}\n")
            self.topics_text.insert(tk.END, f"Peso médio: {topic_info['weight']:.3f}\n\n")
        
        # Display visualization
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()
        
        canvas = FigureCanvasTkAgg(fig, self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def cleanup_analysis(self):
        self.status_label.config(text="Análise concluída!")
        self.run_button.config(state='normal')

if __name__ == "__main__":
    root = tk.Tk()
    app = TopicModelingGUI(root)
    root.mainloop()

