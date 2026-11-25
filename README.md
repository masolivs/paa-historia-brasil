# PAA — Busca Semântica e RAG Local

Projeto da disciplina **Projeto e Análise de Algoritmos (PAA 2/2026)**, da Universidade de Brasília (UnB). A aplicação permite fazer perguntas em linguagem natural em uma interface de chat, recuperar trechos relevantes por busca semântica com embeddings densos (`intfloat/multilingual-e5-small`) e índice **HNSW** (`hnswlib`), gerar respostas fundamentadas com uma LLM local via **Ollama** (`qwen2.5:3b`) e consultar as métricas de latência e fontes citadas.

## Integrantes

| Nome | Matrícula |
| :--- | :--- |
| Adriano Lopes de Sousa | 170097960 |
| Breno Back dos Santos Miranda da Silva | 190063980 |
| Gabriel Bismarck Gonçalves Fagundes | 170103323 |
| Mariana Soares Oliveira | 231013663 |
| Mário Augusto Vieira dos Santos | 231035778 |

## Estrutura do repositório

```text
paa-historia-brasil/
├── backend/
│   └── app/
│       ├── main.py                 # API FastAPI conectada ao motor paa_rag
│       ├── config.py               # Configurações via variáveis de ambiente
│       └── schemas.py              # Contratos da API, métricas e citações
├── frontend/                       # Interface de chat (Vue 3 + TypeScript + Tailwind)
├── paa_rag/                        # Motor RAG local (E5 / Word2Vec, HNSW / Cosseno, Ollama e CLI)
├── pipeline/                       # Limpeza de corpus, chunking e geração de embeddings
├── benchmarks/                     # Benchmarks de recuperação, latência e memória
├── evaluation/                     # Avaliação automática das respostas com RAGAS
├── corpus-pilot/                   # Corpus piloto de chunks da Wikipédia (JSONL)
├── embeddings-pilot/               # Manifestos, IDs, metadados e relatórios de busca
├── evaluation-pilot/               # Conjunto de perguntas e gabaritos em português
├── docs/                           # Documentação do pipeline, fontes, licenças e slides
├── tests/                          # Testes unitários do motor de busca
├── .env.example                    # Modelo de configuração de ambiente
└── .gitignore
```

## Pré-requisitos

- **Python 3.11** (ou 3.12).
- **Node.js 18+** e **npm**.
- **Ollama** instalado e ativo na máquina (`http://127.0.0.1:11434`).
- **Microsoft C++ Build Tools** (apenas no Windows, necessário para compilar o `hnswlib`).

## Como rodar o projeto

### 1. Clonar e preparar as variáveis de ambiente

```bash
git clone [https://github.com/masolivs/paa-historia-brasil.git](https://github.com/masolivs/paa-historia-brasil.git)
cd paa-historia-brasil
cp .env.example .env
cp frontend/.env.example frontend/.env
```

No PowerShell:

```powershell
Copy-Item .env.example .env
Copy-Item frontend/.env.example frontend/.env
```

Confirme no `.env` as configurações principais:

```env
OLLAMA_BASE_URL=[http://127.0.0.1:11434](http://127.0.0.1:11434)
OLLAMA_MODEL=qwen2.5:3b
EMBEDDING_MODEL=intfloat/multilingual-e5-small
RAG_MANIFEST=embeddings-pilot/e5-small/manifest.json
RAG_RETRIEVAL=hnsw
MOCK_MODE=false
```

### 2. Baixar a LLM local no Ollama

```bash
ollama pull qwen2.5:3b
ollama list
```

### 3. Instalar dependências do back-end e gerar os vetores do piloto

Na raiz do repositório, crie e ative o ambiente virtual:

```bash
python -m venv .venv-backend
```

- **Windows (PowerShell):** `.\.venv-backend\Scripts\Activate.ps1`
- **Linux/macOS:** `source .venv-backend/bin/activate`

Instale as dependências e gere os vetores do piloto (os arquivos `.npy` não são versionados no Git):

```bash
pip install -r backend/requirements.txt
python pipeline/generate_pilot_embeddings.py --chunks corpus-pilot/chunks.jsonl --output embeddings-pilot/e5-small --model intfloat/multilingual-e5-small --batch-size 32
```

### 4. Executar o back-end (Terminal 1)

Com o ambiente `.venv-backend` ativo na raiz do repositório:

```bash
uvicorn backend.app.main:app --reload --port 8000
```

- Verificação da API: `http://localhost:8000/api/health`
- Documentação interativa: `http://localhost:8000/docs`

> Para testar apenas a interface sem carregar os vetores ou o Ollama, defina `MOCK_MODE=true` no `.env`.

### 5. Executar o front-end (Terminal 2)

Em outro terminal:

```bash
cd frontend
npm install
npm run dev
```

Acesse **`http://localhost:5173`** no navegador.