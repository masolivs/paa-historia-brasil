# Pipeline, métricas e terminologia

## Fluxo do sistema

```text
pergunta em português
  -> representação vetorial da pergunta (E5 ou Word2Vec Average)
  -> busca dos k chunks (cosseno exato ou HNSW)
  -> contexto recuperado e metadados (título e URL)
  -> prompt fundamentado no contexto
  -> inferência da LLM local via Ollama
  -> resposta, citações e latências em texto ou JSON
```

O programa não faz fine-tuning da E5 nem da LLM. A E5 pré-treinada converte
texto em embeddings; Ollama serve a LLM local para inferência; o Word2Vec é
treinado separadamente no piloto para aprender vetores de palavras.

## Representações e busca

### Sentence Embeddings com E5

`intfloat/multilingual-e5-small` codifica cada chunk como vetor float32 de 384
dimensões. O corpus usa o prefixo `passage: ` e as perguntas usam `query: `.
Os vetores são normalizados para que o produto interno corresponda à
similaridade de cosseno.

### Word2Vec Average

Word2Vec aprende vetores de palavras por contexto. No piloto, usa skip-gram
(`sg=1`), dimensão 100, janela 5, frequência mínima 5, cinco épocas e 12
workers. Para representar um chunk ou uma pergunta, o programa calcula a média
dos vetores das palavras conhecidas e normaliza o resultado. Esse baseline não
modela o contexto de uma palavra tão bem quanto embeddings de sentença.

### Cosseno exato

Calcula a similaridade entre a pergunta e todos os N vetores e ordena os
resultados. Custo de consulta aproximadamente O(Nd), para N vetores de dimensão
d; memória adicional de ordenação/score é O(N). É referência exata para medir
recall, mas cresce linearmente com o número de vetores.

### HNSW

HNSW constrói um grafo de vizinhança para busca aproximada. O custo típico de
consulta é sublinear e depende dos parâmetros e dos dados, mas não há garantia
de tempo logarítmico no pior caso. O parâmetro `ef` controla o equilíbrio entre
recall e velocidade durante a consulta; `M` e `ef_construction` afetam a
construção e o tamanho/qualidade do índice. O benchmark atual usa `M=32`,
`ef_construction=200` e `ef=200`.

## Métricas

- **Recall@k**: proporção dos chunks relevantes encontrados entre os primeiros
  k resultados.
- **MRR**: média do inverso da posição do primeiro resultado relevante.
- **MAP**: média da precisão calculada nas posições dos resultados relevantes.
- **Latência média e p95**: tempo típico e percentil 95 observado.
- **Context precision/recall (RAGAS)**: precisão dos contextos recuperados e
  cobertura do conteúdo de referência.
- **Faithfulness (RAGAS)**: quanto das afirmações da resposta é apoiado pelos
  contextos.
- **Answer relevancy/correctness (RAGAS)**: relevância para a pergunta e
  correção em relação à resposta de referência.

RAGAS é avaliação automática baseada em LLM e não substitui inspeção humana.
Use a mesma LLM avaliadora e o mesmo conjunto de perguntas ao comparar os
métodos. As respostas curtas em `evaluation-pilot/expected-answers-pt.jsonl`
devem ser revisadas por uma pessoa.

Os tempos devolvidos pelo CLI separam:

- `startup_ms`: carregar dados, modelo de embedding e índice;
- `embedding`: converter a pergunta em vetor;
- `retrieval`: buscar chunks;
- `generation`: tempo de resposta da LLM pelo Ollama;
- `total`: soma das etapas por pergunta, excluindo inicialização.

## Volume e limitações experimentais

O manifesto do corpus completo registra 9.075.006.885 bytes de texto, que
correspondem a 8,451759 GiB ou cerca de 9,075 GB decimais. Portanto, o corpus
não atinge o mínimo de 10 GB de texto puro indicado no enunciado.

O projeto mantém esse corpus porque coletar, armazenar, limpar e revalidar dados
adicionais exigiria espaço e tempo que não estavam disponíveis nesta etapa.
Expansão pode exigir novos dumps, espaço para originais e intermediários,
processamento, deduplicação e nova validação. Essa limitação deve ser declarada
no resultado; os 8,45 GiB não devem ser apresentados como 10 GB.

Para gerar vetores a partir dos JSONL existentes e passar um manifesto externo
ao CLI, consulte as instruções executáveis na seção
[Gerar embeddings a partir dos dados existentes](../README.md#gerar-embeddings-a-partir-dos-dados-existentes).

Os embeddings e benchmarks foram limitados a 100.000 chunks por recursos locais.
O corpus completo tem 7.082.981 chunks. E5 exige inferência neural em cada
chunk; extrapolar o custo do piloto para milhões de chunks seria de muitos dias
em CPU, além de precisar de mais armazenamento para vetores e índice. Assim,
os resultados representam um piloto, não desempenho medido na base completa.

Há ainda limitações metodológicas:

- 50 perguntas somente em português não medem adequadamente o desempenho
  multilíngue da base.
- Uma única anotação relevante por pergunta pode subestimar resultados
  semanticamente corretos de outros chunks.
- Resultados RAGAS dependem do modelo avaliador e das respostas de referência.
- O tempo da LLM local varia conforme hardware, tamanho de contexto e carga.

## Bibliotecas e licenças

| Componente | Uso | Licença indicada pelo projeto |
|---|---|---|
| Python | scripts e CLI | Python Software Foundation License |
| sentence-transformers | embeddings E5 | Apache-2.0 |
| Transformers | modelo/tokenizador E5 | Apache-2.0 |
| SentencePiece | tokenização | Apache-2.0 |
| NumPy | vetores e operações numéricas | BSD-3-Clause |
| hnswlib | índice aproximado HNSW | Apache-2.0 |
| Gensim | treinamento Word2Vec | LGPL-2.1 |
| Ollama | servir modelo local | MIT |
| Qwen2.5 3B | geração de resposta | Apache-2.0 conforme licença do modelo |
| RAGAS | avaliação RAG | Apache-2.0 |
| Hugging Face Datasets | estrutura dos dados de avaliação | Apache-2.0 |
| LangChain Ollama | adaptador de Ollama para RAGAS | MIT |

Licenças transitivas podem diferir. Verifique os metadados da versão instalada e
as condições do modelo antes da redistribuição.

## Glossário

- **Embedding**: vetor numérico que representa texto.
- **Sentence Embedding**: representação vetorial de uma frase ou chunk inteiro.
- **Word2Vec Average**: média dos vetores Word2Vec das palavras do texto.
- **Similaridade de cosseno**: medida angular de proximidade entre vetores.
- **HNSW**: índice de grafo para busca aproximada de vizinhos vetoriais.
- **RAG**: recuperação de trechos que são fornecidos à LLM como contexto.
- **LLM**: modelo de linguagem que gera a resposta; neste projeto roda localmente.
- **Inferência**: uso de um modelo já treinado para produzir embeddings ou texto.
- **Treinamento**: ajuste/aprendizado de parâmetros; ocorre para Word2Vec, não
  para E5 ou para a LLM deste protótipo.
