import requests
from bs4 import BeautifulSoup

def get_text_from_url(url):
    # Faz a solicitação HTTP para a URL
    response = requests.get(url)

    # Verifica se a solicitação foi bem-sucedida
    if response.status_code == 200:
        # Analisa o conteúdo HTML da página
        soup = BeautifulSoup(response.content, "html.parser")

        # Remove scripts e styles do conteúdo
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()

        # Obtém o texto puro da página
        text = soup.get_text()

        # Remove espaços em branco excessivos
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = "\n".join(chunk for chunk in chunks if chunk)

        return text
    else:
        return f"Erro ao acessar a página: {response.status_code}"