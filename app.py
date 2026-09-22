import sys
import os

# Remove o diretório atual do topo da busca de caminhos do Python
diretorio_atual = os.path.dirname(os.path.abspath(__file__))
if diretorio_atual in sys.path:
    sys.path.remove(diretorio_atual)
sys.path.append(diretorio_atual)

import time
import tracemalloc
import re
import requests
from bs4 import BeautifulSoup
import streamlit as st
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_community.callbacks.manager import get_openai_callback
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

VECTOR_STORE = None

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Sandbox de Engenharia de Requisitos Ágeis - UFF",
    page_icon="",
    layout="wide"
)

# ---------------------------------------------------------
# CONFIGURAÇÃO SEGURA DA API KEY
# ---------------------------------------------------------
load_dotenv(override=True)

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    try:
        if "OPENAI_API_KEY" in st.secrets and st.secrets["OPENAI_API_KEY"]:
            api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass

if not api_key:
    st.error(" Chave da OpenAI não configurada. Adicione nos Secrets do Streamlit Cloud ou no arquivo .env local.")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key)

def get_response_from_openai(messages):
    return llm.invoke(messages)

def get_text_from_url(url: str) -> str:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()
            
        text = soup.get_text(separator=' ')
        
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        cleaned_text = '\n'.join(chunk for chunk in chunks if chunk)
        
        if not cleaned_text or len(cleaned_text) < 30:
            return f"Aviso: Não foi possível extrair um texto significativo da URL ({url}). Certifique-se de que o documento está publicado na Web."
            
        return cleaned_text
        
    except Exception as e:
        return f"Erro ao extrair conteúdo da URL ({url}): {str(e)}"


def recalcular_metricas_markdown(texto_metricas: str) -> str:
    linhas = texto_metricas.strip().split('\n')
    
    notas_qus = []
    notas_invest = []
    
    for linha in linhas:
        # Regex flexível para capturar variações de identificadores das USs
        if "|" in linha and re.search(r'US-?\d+', linha, re.IGNORECASE):
            colunas = [c.strip() for c in linha.split('|')]
            if colunas and colunas[0] == '':
                colunas.pop(0)
            if colunas and colunas[-1] == '':
                colunas.pop()
                
            if len(colunas) >= 4:
                match_qus = re.search(r'(\d+)\s*/\s*7', colunas[1]) or re.search(r'(\d+)', colunas[1])
                match_invest = re.search(r'(\d+)\s*/\s*6', colunas[3]) or re.search(r'(\d+)', colunas[3])
                
                if match_qus:
                    notas_qus.append(int(match_qus.group(1)))
                if match_invest:
                    notas_invest.append(int(match_invest.group(1)))

    if notas_qus and notas_invest:
        total_qus = sum(notas_qus)
        max_qus = len(notas_qus) * 7
        total_invest = sum(notas_invest)
        max_invest = len(notas_invest) * 6
        
        texto_metricas = re.sub(
            r'- MÉTRICAS QUS:.*', 
            f'- MÉTRICAS QUS: {total_qus} / {max_qus}', 
            texto_metricas
        )
        texto_metricas = re.sub(
            r'- MÉTRICAS INVEST:.*', 
            f'- MÉTRICAS INVEST: {total_invest} / {max_invest}', 
            texto_metricas
        )
        
    return texto_metricas

# --- FERRAMENTAS DO AGENTE ---
@tool
def rag_indexing_tool(url: str) -> str:
    """
    Acessa a URL, realiza o chunking do documento e cria a base vetorial FAISS em memória.
    """
    global VECTOR_STORE
    raw_text = get_text_from_url(url)
    
    if raw_text.startswith("Erro") or raw_text.startswith("Aviso"):
        return raw_text

    # Fragmentação adaptativa
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", ";", " "]
    )
    docs = text_splitter.create_documents([raw_text])
    
    # Geração do Vector Store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    VECTOR_STORE = FAISS.from_documents(docs, embeddings)
    
    return f"Indexação RAG concluída com sucesso. Total de {len(docs)} fragmentos processados."

@tool
def normative_analysis_tool(query_escopo: str) -> str:
    """
    Realiza busca vetorial no documento para identificar lacunas normativas, 
    regras ausentes ou contradições antes de gerar os requisitos.
    """
    global VECTOR_STORE
    if not VECTOR_STORE:
        return "Erro: Base vetorial não inicializada. Execute rag_indexing_tool primeiro."

    # Recuperação dos trechos mais relevantes do escopo
    docs_relevantes = VECTOR_STORE.similarity_search(query_escopo, k=6)
    contexto_recuperado = "\n---\n".join([d.page_content for d in docs_relevantes])

    messages = [
        SystemMessage(content="""
Você é um Engenheiro de Requisitos especialista em Análise Normativa e Auditoria Primária.
Examine o texto recuperado do documento de escopo e identifique:
1. REGRAS OMISSAS OU INCOMPLETAS (Ex: regras de validação ausentes, fluxos alternativos não descritos).
2. DUVIDAS / LACUNAS que precisam de confirmação do analista.
3. CONTRADIÇÕES internas.

Formato da resposta:
# ANÁLISE NORMATIVA E DIAGNÓSTICO DE LACUNAS
## 1. REGRAS IDENTIFICADAS E VALIDADAS
- [Regras confirmadas no texto]
## 2. LACUNAS E OMISSÕES DETECTADAS
- [Ponto Omisso]: [Explicação da ausência e impacto na regra de negócio]
## 3. PERGUNTAS DE REFINAMENTO SUGERIDAS
- [Pergunta objetiva para o Analista/PO sanar a lacuna]
"""),
        HumanMessage(content=f"Contexto Recuperado via RAG:\n{contexto_recuperado}")
    ]
    return get_response_from_openai(messages).content

@tool
def user_story_rag_generation_tool(payload_contexto: str) -> str:
    """
    Gera um Backlog de Histórias de Usuário exaustivo e atômico, 
    garantindo cobertura completa de todos os módulos e regras do documento.
    """
    global VECTOR_STORE
    if not VECTOR_STORE:
        return "Erro: Base vetorial não inicializada."

    # Busca abrangente para recuperar todos os módulos e regras de negócio
    docs_relevantes = VECTOR_STORE.similarity_search(
        "Módulos, funcionalidades, regras de negócio, personas e integrações", 
        k=15
    )
    contexto_completo = "\n---\n".join([f"[Trecho {i+1}]: {d.page_content}" for i, d in enumerate(docs_relevantes)])

    messages = [
    SystemMessage(content="""
Você é um Engenheiro de Requisitos Sênior especializado em detalhamento fino de software industrial.
Sua missão é gerar um Backlog de Histórias de Usuário extremamente ESPECÍFICO, DETALHADO e com CRITÉRIOS DE ACEITAÇÃO RICOS baseados no contexto fornecido.

EVITE GENERALISMOS:
- PROIBIDO criar critérios de aceitação genéricos do tipo "O sistema deve permitir X".
- CADA História de Usuário DEVE possuir de 2 a 4 Critérios de Aceitação (CA) detalhados contendo:
  1. Campos obrigatórios/opcionais envolvidos.
  2. Regras de validação, bloqueios ou pré-condições.
  3. Formatos esperados ou comportamentos do sistema em caso de sucesso/erro.

FORMATO OBRIGATÓRIO:

US-[NÚMERO] - [NOME ESPECÍFICO DA FUNCIONALIDADE]
Como <Persona/Ator>,
quero <Ação Única e Concreta com contexto do negócio>,
para que <Benefício Direto e Métrica/Impacto do negócio>.

Critérios de Aceitação:
- CA : [Detalhamento concreto da regra, campo, validação ou comportamento] (Origem: Trecho X)
[Adicione quantas linhas de CA forem necessárias para cobrir todos os cenários da história]
"""),
        HumanMessage(content=f"Contexto do Documento de Escopo:\n{contexto_completo}")
    ]
    return get_response_from_openai(messages).content

@tool
def semantic_consistency_tool(payload: str) -> str:
    """
    Auditoria semântica e verificação de aderência de escopo.
    O 'payload' deve conter o texto completo do CONTEXTO e das HISTÓRIAS concatenados.
    """
    messages = [
        SystemMessage(content="""
Você é um Auditor Semântico de Requisitos.
Sua função é verificar se CADA User Story PERTENCE ao escopo do projeto.
Uma história pode ser perfeitamente escrita (QUS 7/7), mas ser REPROVADA SEMANTICAMENTE por não constar no escopo.

Para cada User Story responda:
US-XX
Status: [APROVADO SEMANTICAMENTE] ou [REPROVADO SEMANTICAMENTE]
Justificativa: Explicar concisamente se a funcionalidade existe no documento de escopo.
"""),
        HumanMessage(content=f"DADOS PARA ANÁLISE SEMÂNTICA:\n\n{payload}")
    ]
    return get_response_from_openai(messages).content

@tool
def requirements_coverage_tool(payload: str) -> str:
    """
    Matriz de Rastreabilidade entre regras de escopo e User Stories.
    O 'payload' deve conter o texto completo do CONTEXTO e das HISTÓRIAS concatenados.
    """
    messages = [
        SystemMessage(content="Retorne SOMENTE um JSON válido com a matriz de rastreabilidade."),
        HumanMessage(content=f"DADOS PARA RASTREABILIDADE:\n\n{payload}\n\nFormat: [{{\"Regra\":\"...\", \"UserStory\":\"US-XX\", \"Observacao\":\"...\"}}]")
    ]
    return get_response_from_openai(messages).content

@tool
def quality_assessment_tool(user_stories: str) -> str:
    """
    Avaliação estritamente estrutural/sintática no framework QUS (7 dimensões).
    """
    messages = [
        SystemMessage(content="""Você é um auditor do framework QUS (Quality of User Stories).
Avalie cada história em 7 critérios binários (1/0):
1. Bem-formada (Well-formed)
2. Atômica (Atomic)
3. Conceitualmente Sólida (Conceptually Sound)
4. Inambígua (Unambiguous)
5. Mínima (Minimal)
6. Sentença Completa (Full Sentence)
7. Estimável (Estimable)

SE A NOTA FOR MENOR QUE 7, VOCÊ É OBRIGADO A NOMEAR EXATAMENTE QUAIS CRITÉRIOS FALHARAM.
Exemplo para nota 6/7: "Falha: Atômica". NUNCA omita a justificativa da perda de ponto."""),
        HumanMessage(content=f"Avalie:\n{user_stories}")
    ]
    return get_response_from_openai(messages).content

@tool
def invest_assessment_tool(user_stories: str) -> str:
    """
    Avaliação de viabilidade ágil no acrônimo INVEST (6 dimensões).
    """
    messages = [
        SystemMessage(content="""Você é um Agile Coach especialista em INVEST.
Avalie cada história em 6 critérios binários (1/0):
1. Independent
2. Negotiable
3. Valuable
4. Estimable
5. Small
6. Testable

SE A NOTA FOR MENOR QUE 6, VOCÊ É OBRIGADO A NOMEAR EXATAMENTE QUAIS CRITÉRIOS FALHARAM.
Exemplo para nota 5/6: "Falha: Small". NUNCA omita a justificativa da perda de ponto."""),
        HumanMessage(content=f"Avalie:\n{user_stories}")
    ]
    return get_response_from_openai(messages).content

# --- CONFIGURAÇÃO DO AGENTE E PROMPT ---
toolkit = [
    rag_indexing_tool,
    normative_analysis_tool,
    user_story_rag_generation_tool,
    semantic_consistency_tool, 
    requirements_coverage_tool, 
    quality_assessment_tool, 
    invest_assessment_tool
]

prompt = ChatPromptTemplate.from_messages([
    ("system", """
Você é um Engenheiro de Software automatizado especialista em Engenharia de Requisitos com suporte a RAG.
Instruções estritas do fluxo com ancoragem e análise de lacunas:

1. Execute 'rag_indexing_tool' enviando a URL informada para vetorizar o documento.
2. Execute 'normative_analysis_tool' com a busca das principais regras e personas do projeto para gerar o relatório de lacunas.
3. Execute 'user_story_rag_generation_tool' com base nas personas/módulos mapeados para gerar o backlog com rastreabilidade explícita aos trechos.
4. Execute 'semantic_consistency_tool' concatenando o relatório de análise normativa e as histórias no parâmetro 'payload'.
5. Execute 'requirements_coverage_tool' concatenando a análise normativa e as histórias no parâmetro 'payload'.
6. Execute 'quality_assessment_tool' enviando as histórias geradas.
7. Execute 'invest_assessment_tool' enviando as histórias geradas.

Organize a saída utilizando estritamente as tags de seção e esquemas definidos abaixo:

[SECAO_ANALISE_NORMATIVA]
Exiba o resultado completo gerado pela ferramenta 'normative_analysis_tool', destacando lacunas e omissões encontradas.

[SECAO_US]
Apresente o Backlog Completo das User Stories ancoradas com rastreabilidade de trechos gerado pela 'user_story_rag_generation_tool'.

[SECAO_METRICAS]
#### 1. Consolidação Quantitativa das Métricas Sintáticas e Estruturais
- MÉTRICAS QUS: [SOMA_INTEIRA_NOTAS_QUS] / [TOTAL_HISTORIAS * 7]
- MÉTRICAS INVEST: [SOMA_INTEIRA_NOTAS_INVEST] / [TOTAL_HISTORIAS * 6]

*Regras de Consolidação:*
- O numerador DEVE ser a soma exata dos pontos obtidos por todas as histórias (exemplo: se forem 3 histórias com notas 6, 7 e 6, a soma é 19).
- NÃO exiba divisão com casas decimais ou porcentagens na linha da consolidação. Use estritamente o formato `X / Y`.

#### 2. Relatório de Auditoria Estrutural (QUS e INVEST)

A tabela abaixo DEVE seguir este cabeçalho e formato padronizado de 1 linha por User Story (NÃO crie tabelas separadas por critérios individuais):

| User Story | Nota QUS | Falhas QUS | Nota INVEST | Falhas INVEST |
| :--- | :--- | :--- | :--- | :--- |
| US-001 | 6/7 | Atômica | 6/6 | Nenhuma |
| US-002 | 7/7 | Nenhuma | 6/6 | Nenhuma |

*Instruções para preenchimento da Tabela:*
- 'Nota QUS': Soma dos critérios QUS atendidos pela história sobre 7 (ex: 6/7).
- 'Falhas QUS': Nome do critério QUS com nota 0 (ou 'Nenhuma' se nota for 7/7).
- 'Nota INVEST': Soma dos critérios INVEST atendidos pela história sobre 6 (ex: 5/6).
- 'Falhas INVEST': Nome do critério INVEST com nota 0 (ou 'Nenhuma' se nota for 6/6).

[SECAO_AVALIADOR]
#### 1. Relatório de Validação de Domínio e Nexo Semântico
Resultado da 'semantic_consistency_tool'.

#### 2. Matriz de Cobertura de Requisitos
Resultado da 'requirements_coverage_tool'.
"""),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_tools_agent(llm, toolkit, prompt)
agent_executor = AgentExecutor(agent=agent, tools=toolkit, verbose=False)

# --- INTERFACE DO STREAMLIT ---

st.title("Geração de Histórias de Usuário - UFF")
st.subheader("Agentes Multi-Ferramentas")
st.markdown("---")

url_padrao = "https://docs.google.com/document/d/e/2PACX-1vTwj4Yh9UVPzqEpHJMprp875O7bW6XRQek_JNl-1ZxriLWvXvWInIxlxaYY4-yTRRTvxNIvUSPkuFbm/pub"
url_documento = st.text_input("Cole aqui a URL pública do documento de escopo (Google Docs publicado na Web ou site):", value=url_padrao)

if st.button(" Iniciar Pipeline de IA", use_container_width=True):
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
                
                st.success("Processamento Concluído com Sucesso!")
                
                output_total = result["output"]
                
                conteudo_us = ""
                conteudo_metricas = ""
                conteudo_avaliador = ""
                conteudo_normativa = ""
                
                if "[SECAO_ANALISE_NORMATIVA]" in output_total:
                    p_norm = output_total.split("[SECAO_ANALISE_NORMATIVA]")[1]
                    conteudo_normativa = p_norm.split("[SECAO_US]")[0] if "[SECAO_US]" in p_norm else p_norm

                if "[SECAO_US]" in output_total:
                    p_us = output_total.split("[SECAO_US]")[1]
                    conteudo_us = p_us.split("[SECAO_METRICAS]")[0] if "[SECAO_METRICAS]" in p_us else p_us
                
                if "[SECAO_METRICAS]" in output_total:
                    p_met = output_total.split("[SECAO_METRICAS]")[1]
                    conteudo_metricas = p_met.split("[SECAO_AVALIADOR]")[0] if "[SECAO_AVALIADOR]" in p_met else p_met
                
                if "[SECAO_AVALIADOR]" in output_total:
                    conteudo_avaliador = output_total.split("[SECAO_AVALIADOR]")[1]

                if conteudo_metricas:
                    conteudo_metricas = recalcular_metricas_markdown(conteudo_metricas)

                tab0, tab1, tab2, tab3, tab4 = st.tabs([
                    " Análise Normativa & Lacunas",
                    " Backlog Ancorado (RAG)", 
                    " Métricas de Qualidade (QUS/INVEST)", 
                    " Validação Semântica & Cobertura", 
                    " Desempenho & Recursos"
                ])
                
                with tab0:
                    st.header(" Análise Normativa e Detecção de Omissões no Documento")
                    st.markdown(conteudo_normativa.strip())
                with tab1:
                    st.header(" Backlog de User Stories Gerado")
                    st.markdown(conteudo_us.strip())
                    
                with tab2:
                    st.header(" Métricas e Auditoria Qualitativa (Estrutura e Sintaxe)")
                    st.markdown(conteudo_metricas.strip())
                    
                with tab3:
                    st.header(" Relatório do Avaliador (Aderência de Escopo e Semântica)")
                    st.markdown(
                        """
                        <style>
                        section[data-testid="stMarkdownContainer"] table { width: 100%; }
                        section[data-testid="stMarkdownContainer"] th { background-color: #1E293B; color: white; text-align: center; }
                        section[data-testid="stMarkdownContainer"] td { vertical-align: top; white-space: pre-wrap; }
                        </style>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown(conteudo_avaliador.strip())
                    
                with tab4:
                    st.header("⚡ Desempenho Técnico do Pipeline")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(label="⏱ Tempo Total de Execução", value=f"{tempo_final - tempo_inicial:.2f} s")
                        st.metric(label=" Consumo de Memória de Pico", value=f"{memoria_pico / (1024 * 1024):.2f} MB")
                    
                    with col2:
                        st.metric(label=" Tokens de Entrada (Prompt)", value=f"{cb.prompt_tokens}")
                        st.metric(label=" Tokens de Saída (Completion)", value=f"{cb.completion_tokens}")
                    
                    with col3:
                        st.metric(label=" Total de Tokens", value=f"{cb.total_tokens}")
                        st.metric(label=" Custo Real (gpt-4o-mini)", value=f"${custo_real_calculado:.5f} USD")
                        
            except Exception as e:
                st.error(f"Ocorreu um erro durante a execução do agente: {e}")