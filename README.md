# Company Help Assistant (Weeks 11 and 12 Capstone)

A chat app that answers questions about shipping, refunds, and account help. It searches the documents in `docs/`, uses a calculator tool for arithmetic, and shows the sources under each answer. It uses Groq's Llama model.

## Files

| File               | Purpose                                                     |
| ------------------ | ----------------------------------------------------------- |
| `app.py`           | Streamlit chat interface                                    |
| `agent.py`         | Agent that decides when to search documents or calculate    |
| `tools.py`         | Calculator and store info lookup                            |
| `rag.py`           | Groq client setup and answer generation                     |
| `retriever.py`     | Chunking, embeddings, and vector search                     |
| `config.py`        | Loads the API key from the environment or Streamlit secrets |
| `docs/`            | Policy documents the assistant answers from                 |
| `requirements.txt` | Python packages                                             |

## Run locally

1. Activate your virtual environment, then install the packages:

   ```powershell
   pip install -r requirements.txt
   ```

2. Set your Groq API key (never put it in the code):
   - Windows PowerShell: `$env:GROQ_API_KEY="your-key-here"`
   - macOS / Linux: `export GROQ_API_KEY="your-key-here"`

3. Start the app:

   ```powershell
   streamlit run app.py
   ```

   Open the address Streamlit prints (usually http://localhost:8501).

## Deploy to Streamlit Community Cloud

1. Push this folder to a GitHub repository. Make sure `.gitignore` is in place so no key or `venv/` is uploaded.
2. Go to share.streamlit.io, choose "New app", select the repository, and set the main file to `app.py`.
3. Open "Advanced settings" and add this secret:

   ```toml
   GROQ_API_KEY = "your-key-here"
   ```

4. Click Deploy.

## Test checklist

- [ ] "How many days do I have to return an item?" gets a sourced answer from the refund policy
- [ ] "What is 20% of 85 dollars?" uses the calculator and returns 17
- [ ] "Do you ship internationally?" gets the "could not find" message


## Known limits

- The first start downloads the embedding model, which can take a minute.
- Answers are only as good as the documents. Questions outside them should be refused.
- Check the model's answers against the sources, since the model can still make small mistakes.
