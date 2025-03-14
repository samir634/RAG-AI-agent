from agents import Agent, Runner, tool, function_tool
import requests
from bs4 import BeautifulSoup
import os

is_key_set = 'OPENAI_API_KEY' in os.environ

@function_tool
def fetch_webpage_content(url: str) -> str:
    """Fetches and extracts clean text from a webpage."""
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        text_content = "\n".join([p.get_text() for p in soup.find_all("p")])
        return text_content
    else:
        return f"Failed to fetch webpage: {response.status_code}"

def main():

    #gpt 3.5 used here
    agent = Agent(name="Assistant", model="gpt-3.5-turbo", instructions="Use the url provided in the request to primarily generate results for the question", tools=[fetch_webpage_content])

    result = Runner.run_sync(agent, "Was Henry VIII a good person? Here is some information: https://en.wikipedia.org/wiki/Henry_VIII")
    print(result.final_output)


if __name__ == '__main__' and is_key_set:
    main()
else:
    print("Set OPENAI_API_KEY")

