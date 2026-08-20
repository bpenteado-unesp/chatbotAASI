# RAG AASI — Protótipo

Chatbot de dúvidas técnicas sobre AASI, com triagem que impede o bot
de responder sintoma pessoal / pedido de avaliação clínica.

## Passo a passo

1. Instalar dependências:
   ```
   pip install -r requirements.txt --break-system-packages
   ```

2. Criar o arquivo `.streamlit/secrets.toml` na raiz do projeto (mesma
   pasta de onde você roda os scripts) com a sua chave da Gemini API:
   ```toml
   [gemini]
   api_key = "sua_chave_aqui"
   ```
   Pegue a chave em https://aistudio.google.com/app/apikey

   Esse é o mecanismo nativo do Streamlit (`st.secrets`) — funciona tanto
   com `streamlit run` quanto nos scripts avulsos (`testar_local.py`,
   `app.py`), desde que rodados a partir dessa mesma pasta.

   **Importante:** adicione `.streamlit/secrets.toml` ao `.gitignore` —
   nunca suba esse arquivo para um repositório público.

3. Colocar os PDFs em `./data/pdfs`

4. **Pré-baixar o modelo de embeddings** (evita que o download aconteça na
   primeira mensagem de um aluno em produção):
   ```
   python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', cache_folder='./assets')"
   ```
   Isso baixa e guarda o modelo em `./assets/` (não mais no cache padrão
   do usuário). Rode isso uma vez durante o build/deploy do servidor,
   antes de liberar o webhook para uso — assim a primeira pergunta do
   aluno não fica lenta esperando o download.

   Em produção, depois desse passo, é recomendável impedir tentativa de
   download em runtime, para falhar rápido em vez de travar caso a
   internet do servidor não tenha acesso ao Hugging Face:
   ```
   export HF_HUB_OFFLINE=1
   ```
   **Isso vale para servidor próprio/VPS, onde você controla o ambiente
   antes do deploy. NÃO se aplica ao Streamlit Cloud** — lá não tem como
   pré-baixar antes de subir; o download em runtime é esperado e faz parte
   do funcionamento normal (veja a seção "Deploy no Streamlit Cloud" abaixo).

5. Rodar a ingestão (só precisa rodar de novo se os PDFs mudarem):
   ```
   python ingest.py
   ```
   (usa `./data/pdfs` e `./data/chromadb` como padrão; pode sobrescrever
   com `--pdf_dir` e `--db_dir` se precisar)

   **Se você já tinha rodado a ingestão antes de 20/08/2026** (ou de
   qualquer forma, se `rag.py`/`ingest.py` mudaram para usar
   `metadata={"hnsw:space": "cosine"}`), apague a pasta `./data/chromadb`
   antes de rodar de novo — a métrica de distância é fixada na criação
   da coleção e não muda automaticamente:
   ```
   rm -rf ./data/chromadb   # Windows: rd /s /q .\data\chromadb
   python ingest.py
   ```

6. Testar localmente, sem WhatsApp ainda — duas opções:
   - Terminal:
     ```
     python testar_local.py
     ```
   - Interface visual (recomendado para validar com colegas/professores):
     ```
     streamlit run streamlit_app.py
     ```

7. Quando estiver validado, subir o webhook do WhatsApp:
   ```
   uvicorn app:app --reload
   ```
   Expor com `ngrok http 8000` e colocar a URL (+ `/whatsapp`) no
   console da Twilio (Sandbox do WhatsApp ou número aprovado).

## Arquivos

- `config.py` — caminhos padrão (PDFs, ChromaDB, cache do modelo) e leitura da chave da Gemini API via `st.secrets`
- `ingest.py` — lê os PDFs, quebra em chunks, gera embeddings, salva no ChromaDB
- `triage.py` — filtro que bloqueia pergunta de sintoma pessoal antes do RAG
- `rag.py` — busca vetorial + geração de resposta ancorada no material
- `app.py` — webhook FastAPI que integra tudo com o WhatsApp via Twilio
- `testar_local.py` — teste no terminal, sem depender do WhatsApp
- `streamlit_app.py` — interface visual (chat) para testar sem WhatsApp

## Estrutura de pastas

```
.
├── config.py
├── ingest.py
├── rag.py
├── triage.py
├── app.py
├── testar_local.py
├── streamlit_app.py
├── .streamlit/
│   └── secrets.toml      <- você cria (não versionar)
├── data/
│   ├── pdfs/             <- seus PDFs de origem
│   └── chromadb/         <- gerado pelo ingest.py
└── assets/               <- modelo de embeddings baixado (gerado)
```

## Deploy no Streamlit Cloud

O Streamlit Cloud puxa o código direto do GitHub e roda `pip install -r
requirements.txt`. Isso muda como pensar em três coisas que funcionam
diferente de rodar local:

**1. O modelo de embeddings NUNCA deve ser versionado no Git**
(`./assets/`, já no `.gitignore`). GitHub barra arquivo > 100MB sem Git
LFS, e o modelo `paraphrase-multilingual-MiniLM-L12-v2` passa disso
tranquilamente. A solução não é "encolher" ou usar Git LFS — é simplesmente
deixar o código baixar o modelo sozinho, o que ele já faz: a linha
`SentenceTransformer(NOME_MODELO_EMBEDDING, cache_folder=str(CAMINHO_ASSETS))`
em `ingest.py`/`rag.py` baixa do Hugging Face automaticamente se a pasta
`./assets/` não existir ainda.

No Streamlit Cloud, isso significa: **na primeira vez que o container
"acorda"** (após um push novo, ou depois de ficar inativo e dormir — comum
no plano gratuito), a primeira pessoa a abrir o app espera um pouco mais
enquanto o modelo baixa (algumas dezenas de segundos). Depois disso, graças
ao `st.cache_resource` que já configuramos, o modelo fica em memória e todo
mundo que acessar durante aquele período de atividade do container tem
resposta rápida. Quando o container reinicia de novo (novo deploy, ou
voltando do modo hibernado), o ciclo se repete — é esperado, não é bug.

**2. O `data/chromadb` (banco vetorial já processado) DEVE ser
versionado**, ao contrário do modelo. Ele é só os vetores + textos dos
chunks, tipicamente de poucos MB a poucas dezenas de MB — bem abaixo do
limite do GitHub, mesmo com vários PDFs. Versionar ele significa que o
Streamlit Cloud sobe já com a base de conhecimento pronta, sem precisar
rodar `ingest.py` (que reprocessaria e re-embeddaria tudo) toda vez que o
container reinicia — isso seria lento e desnecessário, já que o conteúdo
dos PDFs não muda entre deploys.

Fluxo correto: você roda `python ingest.py` **localmente**, no seu laptop,
sempre que os PDFs mudarem. Depois faz commit da pasta `data/chromadb`
atualizada e dá push. O Streamlit Cloud nunca precisa rodar `ingest.py`
sozinho.

**3. A chave da Gemini API vai no painel do Streamlit Cloud, não no
repositório.** Como `.streamlit/secrets.toml` está no `.gitignore` (nunca
sobe pro GitHub), configure a chave em: no painel do seu app no Streamlit
Cloud → **Settings → Secrets** → cole o mesmo conteúdo TOML:
```toml
[gemini]
api_key = "sua_chave_aqui"
```
O `st.secrets` no código já lê isso automaticamente, seja localmente (do
arquivo) ou na nuvem (do painel) — nenhuma mudança de código necessária.

**Resumo do que sobe pro GitHub:**
| Pasta/arquivo | Sobe pro Git? |
|---|---|
| `*.py` (todo o código) | ✅ Sim |
| `./data/pdfs/*.pdf` | ✅ Sim (fonte original) |
| `./data/chromadb/` | ✅ Sim (processado localmente, versionado) |
| `./assets/` (modelo baixado) | ❌ Não — baixa sozinho em runtime |
| `.streamlit/secrets.toml` | ❌ Não — configurar no painel do Streamlit Cloud |

**Atenção à memória**: o plano gratuito do Streamlit Cloud tem RAM
limitada (historicamente ~1GB). O modelo de embeddings + índice BM25 +
overhead do Streamlit podem chegar perto desse limite dependendo do
volume de PDFs. Se o app cair com erro de memória, esse é o primeiro
suspeito — nesse caso, um modelo de embedding menor seria a solução mais
direta, não um ajuste de código.

## Pontos de atenção

- A triagem tem duas camadas: palavra-chave (rápida, grátis) e
  classificador via LLM (para casos ambíguos). Revise a lista de
  palavras em `triage.py` com quem conhece o público real dos alunos.
- **Citação de fonte**: cada resposta cita a fonte do PDF, a **seção do
  manual** (ex: `seção "7.4.1 Inserir o molde auricular"`) e uma **página
  aproximada**. A seção é a referência principal porque é texto real,
  extraído do próprio conteúdo — sempre confiável. A página é aproximada
  porque, testando em um manual real (GN ReSound), a numeração impressa
  ficou irregular a partir de certo ponto do documento (provavelmente
  duas numerações sobrepostas no design do PDF); por isso ela é só um
  complemento, não a referência principal.
- **Formato do PDF**: o `ingest.py` foi calibrado para manuais de
  fabricante onde cada página do arquivo é, na prática, um "spread" com
  duas páginas impressas lado a lado em duas colunas (comum nesse tipo
  de manual). A extração separa as colunas antes de ler o texto — sem
  isso, frases de seções diferentes se misturam. Se algum PDF do seu
  acervo NÃO for de página dupla/coluna dupla, teste a extração nele
  antes de confiar cegamente — o código assume esse formato.
- **Tabelas**: são detectadas e extraídas como markdown (linhas/colunas
  preservadas), num chunk próprio, e a área da tabela é excluída da
  extração de texto corrido (evita duplicar o mesmo conteúdo de forma
  pior/embaralhada). Limitação conhecida: células que usam ícones em vez
  de texto (ex: notas musicais indicando número de bipes) não são
  capturadas — apenas texto real é extraído.
- **Diagramas com números de legenda** (ex: ilustração do aparelho com
  1-16 apontando peças): o texto das legendas é extraído, mas a ordem
  pode ficar um pouco confusa, porque os números da legenda "vazam" para
  o meio do fluxo de texto (eles são apenas texto solto sobre um desenho,
  sem informação de que fazem parte da imagem). Isso não tem solução
  limpa sem OCR guiado por layout — é uma limitação aceita neste
  protótipo.
- Se a busca não encontrar trechos suficientemente relevantes (limiar de
  similaridade em `rag.py`, parâmetro `distancia_maxima`), ou se o modelo
  julgar que os trechos não respondem à pergunta com segurança, o bot
  responde que não sabe e recomenda procurar o professor — em vez de
  inventar uma resposta.
- Isso é um protótipo. Antes de colocar em produção com alunos de
  verdade, vale registrar logs das perguntas desviadas para auditoria,
  e ter um humano revisando periodicamente se a triagem está pegando
  os casos certos.
- O limiar `distancia_maxima` em `rag.py` precisa ser calibrado com
  perguntas reais depois que os PDFs entrarem — o valor 0.6 é um ponto
  de partida, não um número validado.