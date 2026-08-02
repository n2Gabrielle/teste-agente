import sys
import os

# Remove o diretório atual do topo da busca de caminhos do Python
# Isso impede conflitos de nomes com bibliotecas do LangChain
diretorio_atual = os.path.dirname(os.path.abspath(__file__))
if diretorio_atual in sys.path:
    sys.path.remove(diretorio_atual)
sys.path.append(diretorio_atual)

import time
import tracemalloc
import requests
from bs4 import BeautifulSoup
import streamlit as st

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_community.callbacks.manager import get_openai_callback

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Sandbox de Engenharia de Requisitos Ágeis - UFF",
    page_icon="🤖",
    layout="wide"
)

# CONFIGURAÇÃO DE SEGURANÇA DA API OPENAI
try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    # Chave padrão para desenvolvimento local caso não encontre secrets.toml
    os.environ["OPENAI_API_KEY"] = "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"

# Inicialização do Modelo
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def get_response_from_openai(messages):
    return llm.invoke(messages)

def get_text_from_url(url: str) -> str:
    """
    Acessa uma URL pública (ex: Google Docs publicado na Web ou site),
    baixa o conteúdo HTML e extrai o texto limpo de requisitos.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=12)
        response.raise_for_status()
        
        # Parse do HTML com BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove elementos irrelevantes do HTML
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()
            
        text = soup.get_text(separator=' ')
        
        # Limpeza de espaços e quebras de linha em excesso
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        cleaned_text = '\n'.join(chunk for chunk in chunks if chunk)
        
        if not cleaned_text or len(cleaned_text) < 30:
            return f"Aviso: Não foi possível extrair um texto significativo da URL ({url}). Certifique-se de que o documento está publicado na Web."
            
        return cleaned_text
        
    except Exception as e:
        return f"Erro ao extrair conteúdo da URL ({url}): {str(e)}"

# --- DEFINIÇÃO DAS FERRAMENTAS DO AGENTE ---

@tool
def context_extraction_tool(url: str) -> str:
    """
    Acessa a URL, extrai o texto bruto e gera um artefato de Contexto Estruturado
    (Atores, Regras de Negócio Invioláveis e Escopo Macro).
    """
    raw_context = get_text_from_url(url)
    messages = [
        SystemMessage(content="Você é um Engenheiro de Requisitos Sênior especializado em modelagem de domínio."),
        HumanMessage(content=f"""Analise o texto abaixo e extraia o contexto estruturado:
    ---
    {raw_context}
    ---
    Retorne no formato:
    # CONTEXTO ESTRUTURADO DO PROJETO
    ## 1. ATORES DO SISTEMA
    - [Atores]
    ## 2. REGRAS DE NEGÓCIO E RESTRIÇÕES INVIOLÁVEIS
    - [Regras]
    ## 3. ESCOPO MACRO (FUNCIONALIDADES)
    - [Módulos]
    """)
    ]
    return get_response_from_openai(messages).content

@tool
def user_story_generation_tool(structured_context: str) -> str:
    """
    Recebe o Contexto Estruturado e gera User Stories alinhadas ao escopo,
    evitando adicionar funcionalidades inexistentes.
    """

    messages = [
        SystemMessage(content="""
Você é um Especialista em Engenharia de Requisitos Ágeis.

Sua responsabilidade é gerar um backlog de User Stories seguindo rigorosamente
o escopo fornecido.

REGRAS OBRIGATÓRIAS

1. Gere apenas funcionalidades presentes no escopo.

2. Nunca invente funcionalidades.

3. Nunca invente requisitos técnicos.

4. Nunca acrescente:

- Tempo real
- Dashboard
- Relatórios
- PDF
- Excel
- API
- IA
- Machine Learning
- Notificações
- Push
- SMS
- Email
- Chat
- Aplicativo Mobile

exceto quando estiverem explicitamente presentes no escopo.

5. Cada User Story deve representar UMA única funcionalidade.

6. Utilize o padrão:

Como <ator>,
quero <ação>,
para que <benefício>.

7. Gere exatamente DOIS critérios de aceitação.

8. Os critérios devem apenas detalhar a User Story.

9. Não utilize tecnologias.

10. Não crie requisitos não mencionados.

11. Caso alguma funcionalidade dependa de atualização de dados,
utilize expressões como:

- refletir alterações
- apresentar informações atualizadas
- exibir dados atuais

NUNCA utilize:

- tempo real
- atualização instantânea
- sincronização automática

12. Utilize apenas os atores existentes no contexto.

13. Numere as histórias:

US-01
US-02
...
"""),

        HumanMessage(content=f"""
Contexto Estruturado

-------------------------

{structured_context}

-------------------------

Gere todas as User Stories do projeto.
""")
    ]

    return get_response_from_openai(messages).content

@tool
def semantic_consistency_tool(user_stories: str, structured_context: str) -> str:
    """
    Validação semântica baseada em regras objetivas de Engenharia de Requisitos.
    A ferramenta verifica aderência ao domínio, escopo e stakeholders,
    apontando exatamente qual regra foi violada.
    """

    messages = [
        SystemMessage(content="""
Você é um Auditor de Engenharia de Requisitos especializado em
Validação Semântica de User Stories.

Sua função NÃO é dar opinião.

Sua função é aplicar rigorosamente as regras abaixo.

====================================================
REGRAS DE VALIDAÇÃO
====================================================

Para cada User Story execute exatamente estes passos.

PASSO 1
Verifique se a funcionalidade existe no escopo.

PASSO 2
Verifique se a User Story adiciona funcionalidades inexistentes.

PASSO 3
Verifique se contradiz alguma restrição.

PASSO 4
Verifique se o ator realmente pode executar aquela ação.

PASSO 5
Verifique se os critérios de aceitação continuam dentro do escopo.

====================================================
REPROVE SOMENTE QUANDO
====================================================

1)
A User Story introduzir uma funcionalidade que não aparece no escopo.

Exemplos

- Dashboard
- Relatórios
- Exportação PDF
- Exportação Excel
- IA
- Machine Learning
- Aplicativo Mobile
- Notificações
- Push
- SMS
- Chat
- Integração externa não prevista

----------------------------------------------------

2)
Algum critério de aceitação contradizer uma restrição explícita.

----------------------------------------------------

3)
O ator da User Story não possuir responsabilidade compatível.

----------------------------------------------------

4)
A User Story alterar o objetivo principal do sistema.

====================================================
NÃO REPROVE QUANDO
====================================================

NÃO reprove apenas porque:

- o critério é mais detalhado;

- existe uma forma diferente de implementar;

- existe uma decisão técnica;

- existe uma melhoria de interface;

- o texto está mais específico;

DESDE QUE

essas informações não contradigam o escopo.

====================================================
TEMPO REAL
====================================================

Considere "tempo real" uma violação SOMENTE quando:

- exigir sincronização automática;

- exigir atualização instantânea;

- exigir notificações automáticas;

- exigir push;

- exigir WebSocket;

Se a User Story apenas disser que as informações devem refletir as alterações
realizadas posteriormente, NÃO considere violação.

====================================================
FORMATO DA RESPOSTA
====================================================

Para CADA User Story responda exatamente:

US-XX

Status:
[APROVADO SEMANTICAMENTE]
ou
[REPROVADO SEMANTICAMENTE]

Justificativa:

- explique em poucas linhas.

Caso reprove, informe obrigatoriamente:

Trecho do Escopo Violado:

"...copie exatamente o trecho..."

Critério responsável:

"...texto do critério..."

Não invente violações.

Caso não exista violação,
a User Story deve ser APROVADA.
"""),

        HumanMessage(content=f"""
CONTEXTO ESTRUTURADO

-----------------------------------

{structured_context}

-----------------------------------

USER STORIES

-----------------------------------

{user_stories}

-----------------------------------

Realize a auditoria completa.
""")
    ]

    return get_response_from_openai(messages).content

@tool
def requirements_coverage_tool(user_stories: str, structured_context: str) -> str:
    """
    Gera uma matriz de rastreabilidade entre os requisitos do escopo
    e as User Stories geradas pelo agente.
    """

    messages = [

        SystemMessage(content="""
Você é especialista em rastreabilidade de requisitos.

Retorne SOMENTE JSON válido.

Não utilize markdown.

Não utilize blocos de código.

Não utilize tabelas.

"""),

        HumanMessage(content=f"""

CONTEXTO

{structured_context}

USER STORIES

{user_stories}

Retorne exatamente neste formato:

[
    {{
        "Regra":"Disponibilizar campus e sala",
        "Tipo":"Inclusão",
        "UserStory":"US-01",
        "Observacao":"Relaciona-se diretamente..."
    }}
]

""")

    ]

    return get_response_from_openai(messages).content

@tool
def quality_assessment_tool(user_stories: str) -> str:
    """
    Avalia a qualidade das User Stories geradas usando os 7 critérios do framework QUS.
    """
    messages = [
        SystemMessage(content="Você é um auditor de qualidade de requisitos especialista no framework QUS."),
        HumanMessage(content=f"""
            Analise cada uma das User Stories sob os 7 critérios do framework QUS.
            User Stories para avaliar:
            {user_stories}
            
            Termine com a tag:
            [MÉTRICAS QUS: TOTAL_PONTOS / TOTAL_POSSIVEIS]
        """)
    ]
    return get_response_from_openai(messages).content  

@tool
def invest_assessment_tool(user_stories: str) -> str:
    """
    Avalia o backlog de User Stories utilizando o framework INVEST focado em agilidade.
    """
    messages = [
        SystemMessage(content="Você é um Agile Coach especialista no acrônimo INVEST para histórias de usuário."),
        HumanMessage(content=f"""
            Analise as seguintes User Stories sob os 6 critérios do framework INVEST.
            User Stories para avaliar:
            {user_stories}

            Termine com a tag:
            [MÉTRICAS INVEST: TOTAL_PONTOS / TOTAL_POSSIVEIS]
        """)
    ]
    return get_response_from_openai(messages).content

# --- CONFIGURAÇÃO DO AGENTE E PROMPT ---
toolkit = [
    context_extraction_tool, 
    user_story_generation_tool, 
    semantic_consistency_tool, 
    requirements_coverage_tool, 
    quality_assessment_tool, 
    invest_assessment_tool
]

prompt = ChatPromptTemplate.from_messages([
    ("system", """
Você é um Engenheiro de Software automatizado especialista em Engenharia de Requisitos Ágeis e Validação Avançada de Modelos.
Seu fluxo de trabalho é estrito, ordenado e obrigatório:
    1. Execute a 'context_extraction_tool' passando a URL enviada pelo usuário.
    2. Envie o contexto gerado para a 'user_story_generation_tool'.
    3. Envie as histórias geradas E o contexto estruturado da etapa 1 para a 'semantic_consistency_tool'.
    4. Envie as histórias geradas E o contexto estruturado da etapa 1 para a 'requirements_coverage_tool'.
    5. Envie as histórias geradas para a 'quality_assessment_tool'.
    6. Envie as MESMAS histórias geradas para a 'invest_assessment_tool'.
    
IMPORTANTE: Utilize as tags delimitadoras [SECAO_...] exatamente como indicado abaixo para permitir a segmentação na interface gráfica:

[SECAO_US]
Apresente aqui o Backlog Completo das User Stories geradas, formatadas com seus respectivos identificadores, texto no padrão "Como [Ator], quero [Ação], para que [Benefício]" e Critérios de Aceitação.

[SECAO_METRICAS]
#### 1. Consolidação Quantitativa das Métricas
- [MÉTRICAS QUS: ... ]
- [MÉTRICAS INVEST: ... ]
- [RASTREABILIDADE: ... ]

#### 2. Relatório de Auditoria Simplificado (Métricas Estruturais)
Monte uma tabela Markdown comparativa consolidando os dados das ferramentas QUS e INVEST com as colunas exatas:
| ID da US | Funcionalidade Principal | Avaliação QUS (Total: 7) | Avaliação INVEST (Total: 6) | Critério com Falha detectado |

#### 3. Diagnóstico Técnico dos Resultados
Apresente uma justificativa analítica em tópicos explicando:
- Por que a métrica QUS atingiu o resultado obtido.
- Por que a métrica INVEST variou ou se comportou dessa forma.
- Como o rastreamento ativo de cobertura de requisitos previne falhas de omissão de escopo comuns em geradores baseados em LLM.

[SECAO_AVALIADOR]
#### 1. Relatório de Validação de Domínio e Nexo Semântico
(Apresente a análise semântica detalhada da 'semantic_consistency_tool').

#### 2. Matriz de Cobertura de Requisitos
(Apresente a Matriz de Rastreabilidade gerada pela 'requirements_coverage_tool').
"""),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_tools_agent(llm, toolkit, prompt)
agent_executor = AgentExecutor(agent=agent, tools=toolkit, verbose=False)

# --- INTERFACE DO STREAMLIT ---

st.title("🤖 Sandbox de Engenharia de Requisitos Ágeis - UFF")
st.subheader("Geração e Auditoria de Backlog via Agentes Multi-Ferramentas")
st.markdown("---")

url_padrao = "https://docs.google.com/document/d/e/2PACX-1vTwj4Yh9UVPzqEpHJMprp875O7bW6XRQek_JNl-1ZxriLWvXvWInIxlxaYY4-yTRRTvxNIvUSPkuFbm/pub"
url_documento = st.text_input("Cole aqui a URL pública do documento de escopo (Google Docs publicado na Web ou site):", value=url_padrao)

if st.button("🚀 Iniciar Pipeline de IA", use_container_width=True):
    tracemalloc.start()
    tempo_inicial = time.time()
    
    with st.spinner("O Agente está processando o documento, gerando o backlog e aplicando a auditoria... Aguarde."):
        with get_openai_callback() as cb:
            try:
                result = agent_executor.invoke({
                    "input": f"Execute todo o pipeline de engenharia de requisitos com auditoria semântica, de cobertura, QUS e INVEST para o projeto em: {url_documento}"
                })
                
                tempo_final = time.time()
                memoria_atual, memoria_pico = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                
                custo_entrada = (cb.prompt_tokens / 1000000) * 0.15
                custo_saida = (cb.completion_tokens / 1000000) * 0.60
                custo_real_calculado = custo_entrada + custo_saida
                
                st.success("🎉 Processamento Concluído com Sucesso!")
                
                # --- PARSER DA RESPOSTA EM SEÇÕES ---
                output_total = result["output"]
                
                partes_us = output_total.split("[SECAO_US]")
                partes_metricas = output_total.split("[SECAO_METRICAS]")
                partes_avaliador = output_total.split("[SECAO_AVALIADOR]")
                
                conteudo_us = partes_us[1].split("[SECAO_")[0] if len(partes_us) > 1 else output_total
                conteudo_metricas = partes_metricas[1].split("[SECAO_")[0] if len(partes_metricas) > 1 else ""
                conteudo_avaliador = partes_avaliador[1].split("[SECAO_")[0] if len(partes_avaliador) > 1 else ""

                # --- EXIBIÇÃO ORGANIZADA EM ABAS ---
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📋 Backlog de User Stories", 
                    "📊 Métricas de Qualidade (QUS/INVEST)", 
                    "🔍 Validação Semântica & Cobertura", 
                    "⚡ Desempenho & Recursos"
                ])
                
                with tab1:
                    st.header("📋 Backlog de User Stories Gerado")
                    st.markdown(conteudo_us)
                    
                with tab2:
                    st.header("📊 Métricas e Auditoria Qualitativa")
                    st.markdown(conteudo_metricas)
                    
                with tab3:
                    st.header("🔍 Relatório do Avaliador")

                    st.markdown(
                        """
                        <style>
                        section[data-testid="stMarkdownContainer"] table {
                            width: 100%;
                        }

                        section[data-testid="stMarkdownContainer"] th {
                            background-color: #1E293B;
                            color: white;
                            text-align: center;
                        }

                        section[data-testid="stMarkdownContainer"] td {
                            vertical-align: top;
                            white-space: pre-wrap;
                        }
                        </style>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.markdown(conteudo_avaliador)
    
                    
                with tab4:
                    st.header("⚡ Desempenho Técnico do Pipeline")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(label="⏱️ Tempo Total de Execução", value=f"{tempo_final - tempo_inicial:.2f} s")
                        st.metric(label="💾 Consumo de Memória de Pico", value=f"{memoria_pico / (1024 * 1024):.2f} MB")
                    
                    with col2:
                        st.metric(label="📥 Tokens de Entrada (Prompt)", value=f"{cb.prompt_tokens}")
                        st.metric(label="📤 Tokens de Saída (Completion)", value=f"{cb.completion_tokens}")
                    
                    with col3:
                        st.metric(label="🔢 Total de Tokens", value=f"{cb.total_tokens}")
                        st.metric(label="💵 Custo Real (gpt-4o-mini)", value=f"${custo_real_calculado:.5f} USD")
                        
            except Exception as e:
                st.error(f"Ocorreu um erro durante a execução do agente: {e}")