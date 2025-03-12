from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool, WebsiteSearchTool
import os, warnings, pydantic


def main ():
    
    os.environ.get('OPENAI_API_KEY')
    os.environ.get('SERPER_API_KEY')

    serper = SerperDevTool()
    web = WebsiteSearchTool()
    modelType = LLM(model='gpt-4o', temperature=0.2)

    referenceDoc = WebsiteSearchTool(website='<https://en.wikipedia.org/wiki/Henry_VIII>')
    promptTask = "Was Henry VIII a good person?"

    search = Agent(
        role = "Researcher",
        goal = "Extract insights from provided documentation",
        backstory = "You are being used to evaluate the effectiveness of an AI agent",
        tools=[serper, referenceDoc, web],
        llm=modelType
    )

    job = Task(
        description=promptTask,
        expected_output="What is a good summarization of Henry VIII",
        agent=search
    )

    crew = Crew(
        agents=[search],
        tasks=[job],
        verbose=True,
        process=Process.sequential
    )

    crew.kickoff()

if __name__ == '__main__':
    main()
