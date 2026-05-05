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

# CONFIGURAÇÃO DO AGENTE
toolkit = [documentation_tool]

prompt = ChatPromptTemplate.from_messages([
    ("system", """
    Use suas ferramentas para criar user stories.
    Se você não tiver uma ferramenta para responder, diga que não tem uma ferramenta para isso.
    Retorne apenas as user stories, sem nenhum texto adicional.
    """),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_tools_agent(llm, toolkit, prompt)
agent_executor = AgentExecutor(agent=agent, tools=toolkit, verbose=False)

if __name__ == "__main__":
    result = agent_executor.invoke({"input": "Gere user stories para a documentação em: https://dochttps://docs.google.com/document/d/e/2PACX-1vTwj4Yh9UVPzqEpHJMprp875O7bW6XRQek_JNl-1ZxriLWvXvWInIxlxaYY4-yTRRTvxNIvUSPkuFbm/pub"})
    print(result["output"])