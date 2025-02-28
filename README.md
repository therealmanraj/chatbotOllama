# Steps to Set Up and Run Ollama with Langchain & Streamlit

## 1. Install Ollama

Download and install Ollama from: [https://ollama.com/download](https://ollama.com/download)

Verify that it's installed correctly by running:

```bash
ollama version
```

You should see output like:

```text
ollama 0.x.x
```

---

## 2. Set Up Python Virtual Environment

In your project directory, create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate  # Mac/Linux
```

---

## 3. Install Required Python Packages

With the virtual environment active, install necessary packages:

```bash
pip install streamlit langchain langchain-ollama ollama
```

---

## 4. Start Ollama Server

In a new terminal (outside the virtual environment), start the Ollama server:

```bash
ollama serve
```

Ollama will listen at:

```
http://localhost:11434
```

---

## 5. Pull Required Model(s)

Pull the model you need (e.g., DeepSeek):

```bash
ollama pull deepseek-r1:1.5b
```

You can also pull other models, such as:

```bash
ollama pull llama2
ollama pull mistral
ollama pull gemma:7b
```

Check available models at any time:

```bash
ollama list
```

---

## 6. Run Streamlit App

In your project directory, run:

```bash
streamlit run rag_deep.py
```

This launches your chatbot interface in a browser window.

---

## 7. Troubleshooting (What You Encountered)

| Issue                                | Cause                               | Fix                                |
| ------------------------------------ | ----------------------------------- | ---------------------------------- |
| `zsh: command not found: ollama`     | Ollama not installed or not in PATH | Installed Ollama                   |
| `Connection refused`                 | Ollama server not running           | Ran `ollama serve`                 |
| `model "deepseek-r1:1.5b" not found` | Model not downloaded                | Ran `ollama pull deepseek-r1:1.5b` |

---

## 8. Confirm Successful Flow

- Ollama server is running.
- Required models are downloaded.
- Streamlit app is running.
- AI responses are generated via **Langchain + Ollama**.

---

## Notes

- You **must start Ollama (`ollama serve`) before running your app.**
- Each model only needs to be pulled once.
- Use this to check your models:

```bash
ollama list
```

---

## Example Python Snippet (Optional - Check & Pull Model in Code)

You can add this logic to automatically check and pull models before invoking them:

```python
import subprocess

def ensure_model_downloaded(model_name):
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if model_name not in result.stdout:
        print(f"Model {model_name} not found. Pulling now...")
        subprocess.run(["ollama", "pull", model_name], check=True)

ensure_model_downloaded("deepseek-r1:1.5b")
```

---

Let me know if you want me to save this as a file (`README.md` or `setup_instructions.md`) for you!
