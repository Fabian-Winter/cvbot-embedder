# cvbot-embedder

Ingestion pipeline for a RAG chatbot: it reads local text documents,
splits them semantically into chunks, creates embeddings with AWS Bedrock and
indexes them in a ChromaDB running as a container on AWS Fargate.

## How it works

1. **Load** – all `.txt` and `.md` files from the configured directory.
2. **Format-aware chunking** – Markdown files split by headers and text files
  split by paragraphs with adjacent paragraph overlap.
3. **Metadata parsing** – the `> key: value` block below a heading becomes
   structured chunk metadata (see below).
4. **Token splitting** – only chunks exceeding `MAX_CHUNK_TOKENS` are split
   further with `TokenTextSplitter` (`cl100k_base`).
5. **Indexing** – the target collection is dropped and recreated, then the
   chunks are written in batches. The observed metadata schema is stored in the
   collection metadata so that cvbot-retriever knows which fields it may filter
   on.

## Section metadata

A section may declare metadata in blockquote lines directly below its heading,
one field per line:

```markdown
## ACME Inc.
> type: employer
> status: historic
> location: Berlin
> from: 2011-10
> to: 2022-10

Actual chunk text...
```

- Only the **leading** run of blockquote lines is parsed. A block below an
  `####` heading is ignored, because `####` is not a split boundary.
- Keys are normalized (lowercase, umlauts folded, `Tech-Stack` → `tech_stack`).
- Values are split on `,` and `;`, so `> tech: Java, Maven` is matched by a
  filter for either value. Never put prose into a metadata line.
- Fields cascade downwards: a field set below `#` applies to every section of
  the file until a deeper section overrides it.
- `years` is derived automatically from `from`/`to` (`to: ongoing` runs up to
  the current year) and is the field that makes a question about a single year
  matchable. A manually maintained `years` always wins.
- Malformed lines are skipped with a log entry; a section without any metadata
  is indexed exactly as before.
- `source`, `filename`, `chunk_index` and `h1`–`h3` are reserved and cannot be
  overwritten.
- Metadata is also rendered back into the chunk text as a compact line, so the
  values remain semantically searchable.

## Setup

```bash
source venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.13 and AWS credentials with access to `bedrock:InvokeModel`
for the configured embedding model. Credentials are resolved through the usual
boto3 chain (environment variables, profile, or the EC2 instance profile of
the self-hosted GitHub Actions runner the pipeline runs on).

## Configuration

Configuration is done through environment variables. Every value has a default;
usually only `CHROMA_HOST` needs to be set.

| Variable | Default | Meaning |
| --- | --- | --- |
| `DOCUMENTS_DIR` | `documents` | Local directory holding the documents |
| `CHROMA_HOST` | `localhost` | Hostname of the ChromaDB (Fargate service) |
| `CHROMA_PORT` | `8000` | Port of the ChromaDB |
| `CHROMA_COLLECTION` | `cvbot_documents` | Name of the collection |
| `AWS_REGION` | `eu-central-1` | Region of the Bedrock client |
| `EMBEDDING_MODEL_ID` | `amazon.titan-embed-text-v2:0` | Bedrock model ID; stored in the collection metadata so cvbot-retriever picks the matching model |
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

The `run-pipeline.yml` workflow passes `--chroma-port` from the GitHub Actions
repository variable `CHROMA_PORT`. Keep it in sync with cvbot-infra's
`chroma_port` Terraform variable (default `8000`), or remove the flag from the
workflow to fall back to the matching `Settings` default.

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
  metadata.py      `> key: value` parsing, year derivation, schema collection
  embeddings.py    Bedrock embedding model
  vector_store.py  ChromaDB client, collection, indexing
  pipeline.py      Orchestration
  __main__.py      Command line
```

## Known limitations

- Token counting uses `cl100k_base` as an approximation of the Titan tokenizer.
- ChromaDB is deleted and recreated on every run.
- Metadata blocks are only recognised below `#`, `##` and `###` headings.
