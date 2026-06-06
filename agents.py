import os
from scrapper import get_text_from_url
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool

# CONFIGURAÇÃO GLOBAL
os.environ["OPENAI_API_KEY"] = "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"
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

def get_response_from_openai(messages):
    return llm.invoke(messages)

@tool
def documentation_tool(url: str) -> str:
    """
    Acessa a documentação de uma URL e retorna User Stories. v
    """
    context = get_text_from_url(url)
    messages = [
        SystemMessage(content="You are a software development assistant."),
        HumanMessage(content=f"""Analise o conteúdo da documentação técnica e de negócio extraída:
    ---
    {context}
    ---
    
    Sua tarefa é gerar o conjunto COMPLETO de User Stories necessárias para o desenvolvimento deste sistema do zero (End-to-End). 
    
    Considere as seguintes camadas para garantir que nenhuma funcionalidade seja esquecida:
    1. **Jornada do Usuário Principal**: Desde o primeiro acesso/login até a conclusão do objetivo principal.    
    Importante: Não resuma. Gere todas as histórias que seriam necessárias para compor um Backlog de Produto inicial completo.
    """),
    ]
    response = get_response_from_openai(messages)
    return response.content

@tool
def quality_assessment_tool(user_stories: str) -> str:
    """
    Avalia a qualidade das User Stories usando o framework QUS.
    Atribui 1 ponto por critério atendido e justifica detalhadamente cada decisão.
    """
    messages = [
        SystemMessage(content="Você é um auditor de qualidade de requisitos especialista no framework QUS."),
        HumanMessage(content=f"""
            Analise as seguintes User Stories sob os 7 critérios QUS (cada um vale EXATAMENTE 1 ponto).
            
            Para cada User Story, você deve gerar um relatório contendo:
            1. **Pontuação Total** (X de 7 pontos).
            2. **Análise por Critério**: Liste cada um dos 7 critérios e indique se foi 'Atendido' ou 'Não Atendido'.
            3. **Justificativa (O motivo)**: Para cada critério, explique a razão da nota. 
               - Ex: Se 'Não Atendido em Atômica', explique que a história contém múltiplas funcionalidades.
               - Ex: Se 'Atendido em Bem-formada', confirme que possui Persona, Ação e Benefício.

            Critérios de Avaliação:
            1. Bem-formada | 2. Atômica | 3. Conceitualmente Sólida | 4. Inambígua 
            5. Mínima | 6. Sentença Completa | 7. Estimável.

            User Stories para avaliar:
            {user_stories}

            Ao final do relatório, apresente:
            - O somatório total de pontos de todas as histórias.
            - O cálculo da Taxa de Sucesso: (Soma total / (N de histórias * 7)) * 100.
            - Resultado final em porcentagem.
        """)
    ]
    response = get_response_from_openai(messages)
    return response.content  

# CONFIGURAÇÃO DO AGENTE
toolkit = [documentation_tool, quality_assessment_tool]

prompt = ChatPromptTemplate.from_messages([
    ("system", """
Seu fluxo de trabalho obrigatório é:
    1. Executar a 'documentation_tool' para gerar as User Stories.
    2. Executar a 'quality_assessment_tool' passando as histórias geradas.
    
    ESTRUTURA DE RESPOSTA (Siga estritamente esta ordem no output):
    
    # BACKLOG DE USER STORIES
    (Apresente aqui a lista completa de todas as User Stories geradas, sem interrupções).
    
    # AVALIAÇÃO INDIVIDUAL QUS
    (Para cada User Story da lista acima, apresente a avaliação detalhada da ferramenta de qualidade, 
    incluindo a pontuação X/7 e as justificativas de por que cada critério foi ou não atendido).
    
    # PONTUAÇÃO GERAL DO PROJETO
    (Apresente o somatório total de pontos, o total de histórias analisadas e a Taxa de Sucesso QUS final em %).
    
    Regra: Não adicione textos introdutórios ou conclusões informais. Foque na estrutura técnica acima.
    """),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_tools_agent(llm, toolkit, prompt)
agent_executor = AgentExecutor(agent=agent, tools=toolkit, verbose=False)

if __name__ == "__main__":
    result = agent_executor.invoke({"input": "Gere user stories para a documentação em: https://dochttps://docs.google.com/document/d/e/2PACX-1vTwj4Yh9UVPzqEpHJMprp875O7bW6XRQek_JNl-1ZxriLWvXvWInIxlxaYY4-yTRRTvxNIvUSPkuFbm/pub"})
    print(result["output"])