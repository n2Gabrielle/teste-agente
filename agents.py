import os
import time
import tracemalloc
import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_community.callbacks.manager import get_openai_callback

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Gerador de User Stories - UFF",
    page_icon="🤖",
    layout="wide"
)


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def get_response_from_openai(messages):
    return llm.invoke(messages)

def get_text_from_url(url: str) -> str:
    # Mantendo o mock para o ambiente local/teste
    return "Texto bruto extraído do documento acadêmico da UFF com requisitos sobre centralização de materiais e regras como e-mail @id.uff.br obrigatório."

# --- DEFINIÇÃO DAS FERRAMENTAS ---

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
    Recebe o Contexto Estruturado e gera o Backlog de User Stories formal com critérios de aceitação.
    """
    messages = [
        SystemMessage(content="Você é um Analista de Sistemas Ágil especialista em fatiamento de escopo (Splitting User Stories). Sua função é gerar histórias pequenas, focadas em apenas UMA ação do usuário por frase. Evite verbos amplos como 'gerenciar', 'centralizar' ou 'manter' que embutem múltiplos fluxos (CRUD). Em vez disso, quebre-os em ações menores."),
        HumanMessage(content=f"""Com base no Contexto Estruturado abaixo, gere as User Stories necessárias usando o padrão 'Como [Ator], quero [Ação], para que [Benefício]'. Adicione 2 critérios de aceitação simples por história.
        
        {structured_context}
        """)
    ]
    return get_response_from_openai(messages).content

@tool
def semantic_consistency_tool(user_stories: str, structured_context: str) -> str:
    """
    Avalia se as User Stories geradas fazem sentido lógico, se são semanticamente 
    coerentes e se estão estritamente alinhadas ao contexto e domínio real do projeto.
    """
    messages = [
        SystemMessage(content="""Você é um Engenheiro de Software sênior especialista em Validação Semântica de Requisitos.
Sua função é avaliar o NEXO COGNITIVO, COERÊNCIA SEMÂNTICA e o ALINHAMENTO DE DOMÍNIO."""),
        HumanMessage(content=f"""
            Compare as User Stories geradas com o Contexto Estruturado do Projeto.
            CONTEXTO ESTRUTURADO ORIGINAL:
            ---
            {structured_context}
            ---
            USER STORIES GERADAS:
            ---
            {user_stories}
            ---
            Retorne uma análise para cada US e o veredito [APROVADO SEMANTICAMENTE] ou [REPROVADO SEMANTICAMENTE].
        """)
    ]
    return get_response_from_openai(messages).content

@tool
def requirements_coverage_tool(user_stories: str, structured_context: str) -> str:
    """
    Monta uma Matriz de Rastreabilidade (RTM) e verifica se os requisitos foram cobertos.
    """
    messages = [
        SystemMessage(content="Você é um Engenheiro de Garantia de Qualidade de Requisitos e especialista em Rastreabilidade de Modelos."),
        HumanMessage(content=f"""
            Crie uma Matriz de Rastreabilidade ligando as Regras de Negócio e Escopos ao Backlog.
            CONTEXTO ESTRUTURADO ORIGINAL:
            ---
            {structured_context}
            ---
            USER STORIES GERADAS:
            ---
            {user_stories}
            ---
            Retorne uma tabela Markdown e termine obrigatoriamente com a tag:
            [RASTREABILIDADE: X% COBERTO]
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

# --- CONFIGURAÇÃO DO AGENTE ---
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
    
    ESTRUTURA DE RESPOSTA FINAL (Siga rigorosamente esta estrutura enxuta no output final):
    
    ### 3.4. Resultados e Métricas do Estudo Piloto (Pipeline de Agente com Validação Semântica e Cobertura)

    A execução do pipeline automatizado processou o escopo do projeto da UFF, gerou um backlog de User Stories (US) e aplicou uma tripla camada de auditoria analítica e de cobertura. Os resultados consolidados são apresentados abaixo:

    #### 1. Consolidação Quantitativa das Métricas
    - [MÉTRICAS QUS: ... ]
    - [MÉTRICAS INVEST: ... ]
    - [RASTREABILIDADE: ... ]

    #### 2. Relatório de Validação de Domínio e Nexo Semântico
    (Apresente aqui o resumo do resultado obtido na 'semantic_consistency_tool' de forma resumida).

    #### 3. Matriz de Cobertura de Requisitos
    (Apresente o resultado obtido pela 'requirements_coverage_tool' mostrando o mapeamento de cobertura e se houve algum GAP).

    #### 4. Relatório de Auditoria Simplificado (Métricas Estruturais)
    Monte uma tabela Markdown comparativa consolidando os dados das ferramentas QUS e INVEST com as colunas exatas:
    | ID da US | Funcionalidade Principal | Avaliação QUS (Total: 7) | Avaliação INVEST (Total: 6) | Critério com Falha detectado |

    #### 5. Diagnóstico Técnico dos Resultados
    Apresente uma justificativa analítica curta (em tópicos) explicando:
    - Por que a métrica QUS atingiu o resultado obtido.
    - Por que a métrica INVEST variou ou se comportou dessa forma.
    - Como o rastreamento ativo de cobertura de requisitos previne falhas de omissão de escopo comuns em geradores baseados em LLM.
    
    Termine com uma breve conclusão sobre o papel estratégico e insubstituível do engenheiro humano no refino final do fatiamento do backlog.
    """),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent_executor = create_react_agent(llm, toolkit, state_modifier=prompt)

# --- INTERFACE DO STREAMLIT ---

st.title("Geração de Histórias de Usuário - UFF")
st.subheader("Agente Multi-Ferramentas")
st.markdown("---")

# Input para o link do documento de requisitos
url_padrao = "https://docs.google.com/document/d/e/2PACX-1vTwj4Yh9UVPzqEpHJMprp875O7bW6XRQek_JNl-1ZxriLWvXvWInIxlxaYY4-yTRRTvxNIvUSPkuFbm/pub"
url_documento = st.text_input("Cole aqui a URL pública do documento de escopo (Google Docs ou similar):", value=url_padrao)

if st.button("🚀 Iniciar Pipeline de IA", use_container_width=True):
    # 1. Monitoramento de Hardware
    tracemalloc.start()
    tempo_inicial = time.time()
    
    # 2. Execução
    with st.spinner("O Agente está executando as ferramentas em paralelo e auditando os resultados... Aguarde."):
        with get_openai_callback() as cb:
            try:
                # Execução do pipeline
                result = agent_executor.invoke({
                    "input": f"Execute todo o pipeline de engenharia de requisitos com auditoria semântica, de cobertura, QUS e INVEST para o projeto em: {url_documento}"
                })
                
                tempo_final = time.time()
                memoria_atual, memoria_pico = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                
                # Métricas de Custo Personalizadas
                custo_entrada = (cb.prompt_tokens / 1000000) * 0.15
                custo_saida = (cb.completion_tokens / 1000000) * 0.60
                custo_real_calculado = custo_entrada + custo_saida
                
                st.success("🎉 Processamento Concluído com Sucesso!")
                
                # Exibição do Output Principal do Agente
                st.markdown(result["output"])
                
                st.markdown("---")
                st.subheader("⚙️ Métricas Fisiológicas da Execução (Dados para a sua Pesquisa)")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(label="Tempo Total de Execução", value=f"{tempo_final - tempo_inicial:.2f} s")
                    st.metric(label="Consumo de Memória de Pico", value=f"{memoria_pico / (1024 * 1024):.2f} MB")
                
                with col2:
                    st.metric(label="Tokens de Entrada (Prompt)", value=f"{cb.prompt_tokens}")
                    st.metric(label="Tokens de Saída (Completion)", value=f"{cb.completion_tokens}")
                
                with col3:
                    st.metric(label="Total de Tokens", value=f"{cb.total_tokens}")
                    st.metric(label="Custo Real Calculado (gpt-4o-mini)", value=f"${custo_real_calculado:.5f} USD")
                    
            except Exception as e:
                st.error(f"Ocorreu um erro durante a execução do agente: {e}")