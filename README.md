# cvbot-embedder

Ingestion pipeline for a RAG chatbot: it reads local text documents,
splits them semantically into chunks, creates embeddings with AWS Bedrock and
indexes them in a ChromaDB running as a container on AWS Fargate.

## How it works

1. **Load** – all `.txt` and `.md` files from the configured directory.
2. **Format-aware chunking** – Markdown files split by headers and text files
  split by paragraphs with adjacent paragraph overlap.
3. **Token splitting** – only chunks exceeding `MAX_CHUNK_TOKENS` are split
   further with `TokenTextSplitter` (`cl100k_base`).
4. **Indexing** – the target collection is dropped and recreated, then the
   chunks are written in batches.

## Setup

```bash
source venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.13 and AWS credentials with access to `bedrock:InvokeModel`
for the configured embedding model. Credentials are resolved through the usual
boto3 chain (environment variables, profile, IAM role of the Fargate task).

## Configuration

Configuration is done through environment variables. Every value has a default;
usually only `CHROMA_HOST` needs to be set.

| Variable | Default | Meaning |
| --- | --- | --- |
| `DOCUMENTS_DIR` | `documents` | Local directory holding the documents |
| `CHROMA_HOST` | `localhost` | Hostname of the ChromaDB (Fargate service) |
| `CHROMA_PORT` | `8000` | Port of the ChromaDB |
| `CHROMA_SSL` | `false` | Use HTTPS (`1`, `true`, `yes`, `on`) |
| `CHROMA_AUTH_TOKEN` | – | Optional bearer token for ChromaDB |
| `CHROMA_COLLECTION` | `cvbot_documents` | Name of the collection |
| `AWS_REGION` | `eu-central-1` | Region of the Bedrock client |
| `EMBEDDING_MODEL_ID` | `amazon.titan-embed-text-v2:0` | Bedrock model ID |
| `MAX_CHUNK_TOKENS` | `512` | Maximum number of tokens per chunk |
| `TOKEN_CHUNK_OVERLAP` | `50` | Overlap used when splitting by tokens |
| `BATCH_SIZE` | `50` | Chunks per write to ChromaDB |
| `LOG_LEVEL` | `INFO` | Log verbosity: `DEBUG`, `INFO`, `WARNING` or `ERROR` |

## Usage

```bash
python -m cvbot_embedder
```

Selected values can be overridden on the command line:

```bash
python -m cvbot_embedder \
  --documents-dir documents \
  --collection cvbot_documents \
  --chroma-host chroma.cvbot.internal \
  --chroma-port 8000 \
  --log-level DEBUG
```

The collection is dropped and rebuilt on **every** run. Running the pipeline
twice over the same document set therefore yields the same chunk count.

## Tests

```bash
python -m pytest
```

The tests run without AWS or network access: Bedrock is replaced by a
deterministic embedding model and ChromaDB by test doubles.

## Layout

```
cvbot_embedder/
  config.py        Settings from environment variables
  loader.py        Document loading
  chunking.py      Semantic chunking + token splitting
  embeddings.py    Bedrock embedding model
  vector_store.py  ChromaDB client, collection, indexing
  pipeline.py      Orchestration
  __main__.py      Command line
```

## Known limitations

- Token counting uses `cl100k_base` as an approximation of the Titan tokenizer.
- ChromaDB is deleted and recreated on every run.
