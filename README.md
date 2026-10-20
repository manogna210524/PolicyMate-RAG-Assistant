# College Rules Assistant

Students pick a college policy, read it, and ask questions about it. Answers come only from that policy and name the page they came from.

## How it works

1. **Build (once, and after PDF changes):** each PDF's text is split into chunks of about 2000 characters. Each chunk keeps its page number. Amazon Titan turns each chunk into a vector (a list of numbers that represents its meaning). The vectors are saved in a FAISS index, and the chunk text and page numbers are saved in `chunks.json`.
2. **Ask:** the question becomes a vector. FAISS finds the 5 chunks whose vectors are closest to it. If the closest chunk is not similar enough (the **threshold**), the app says the policy doesn't cover the question and never calls the model.
3. **Answer:** Amazon Nova Pro gets the 5 chunks and the question, and writes an answer from them only, with page numbers.

## What's in the project

| Path | Purpose |
|---|---|
| `app.py` | The Streamlit page: policy picker, PDF viewer, chat |
| `core/config.py` | Model IDs, paths, chunk size, `TOP_K`, `SCORE_THRESHOLD` |
| `core/ingest.py` | Reads PDF text and splits it into chunks with page numbers |
| `core/build_data.py` | Builds the index and chunk files for every policy |
| `core/retrieve.py` | Lists built policies, loads a policy, searches its index |
| `core/answer.py` | Threshold check, prompt, and answer |
| `core/bedrock.py` | Titan embeddings and Nova Pro calls (AWS Bedrock) |
| `data/policies.csv` | Policy list: title, category, version, effective date, owner office, PDF file |
| `data/pdfs/` | The policy PDFs |
| `storage/<policy>/` | Built `index.faiss` and `chunks.json` for each policy |
| `tests/` | Tests (no AWS needed) |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then add your AWS keys
```

The AWS user needs permission to call `bedrock:InvokeModel`. In the Bedrock console, enable access to **Amazon Titan Text Embeddings V2** and **Amazon Nova Pro**.

## Build the policy data

```bash
python -m core.build_data
```

This rebuilds every policy in `data/policies.csv`. It writes the files into `storage/`. Commit `storage/` afterwards so the deployed app can read it.

## Run the app

```bash
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)

1. Push this folder to a GitHub repository, including `storage/` and `data/pdfs/`.
2. On share.streamlit.io, create an app from the repository with main file `app.py`.
3. In the app's **Secrets**, add:
   ```toml
   AWS_ACCESS_KEY_ID = "..."
   AWS_SECRET_ACCESS_KEY = "..."
   AWS_REGION = "us-east-1"
   ```
4. To update a policy, rebuild locally, commit `storage/`, and push.

## Tuning

- `SCORE_THRESHOLD` (in `core/config.py` or `.env`): below this similarity, the app refuses. It starts at 0.30. Raise it if off-topic questions get answers. Lower it if valid questions are refused. Test with real questions before deploying.
- `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`: change only if answers miss the right text.

## Tests

```bash
pytest -q
```

The tests use fake vectors, so they don't call AWS.

## Known limits

- Each page is chunked on its own, so a rule that continues onto the next page is split between two chunks.
- Page numbers are the PDF's page numbers. Printed page numbers in the document may differ.
- Scanned PDFs have no text to read. The build stops with a message for them.
- Policies without a stated date show "not stated".
- The assistant does not make disciplinary or legal decisions. The footer points students to official notices and the office.
- To add a policy, add a row to `data/policies.csv`, add its PDF to `data/pdfs/`, and run the build.
