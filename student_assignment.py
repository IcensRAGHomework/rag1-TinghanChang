import json
import traceback
import requests

from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.output_parsers import (ResponseSchema, StructuredOutputParser)
#hw2
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain import hub
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

def get_llm():
    return AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )

def get_calendarific_data(year:int, month:int) -> str:
    api_key = "jh1uea8ZamUFPpYa3UynZQX96rcBmlTV"
    url = f"https://calendarific.com/api/v2/holidays?&api_key={api_key}&country=tw&year={year}&month={month}"
    response = requests.get(url)
    response = response.json()
    response = response.get('response')
    return response

class GetCalendarific(BaseModel):
    year:int = Field(description="year")
    month:int = Field(description="month")

def question_to_prompt(question):
    examples = [
        {"input":"2024年台灣7月紀念日有哪些?", "output":"""
        {
            "Result": [
                {
                    "date": "2024-07-01",
                    "name": "醫師節"
                },
                {
                    "date": "2024-07-15",
                    "name": "解嚴紀念日"
                }
            ]
        }
         """},
        {"input":"2024年台灣8月紀念日有哪些?", "output":"""
        {
            "Result": [
                {
                    "date": "2024-08-08",
                    "name": "父親節"
                }
            ]
        }
         """},
        {"input":"2024年台灣9月紀念日有哪些?", "output":"""
        {
            "Result": [
                {
                    "date": "2024-09-03",
                    "name": "軍人節"
                },
                {
                    "date": "2024-09-28",
                    "name": "教師節"
                }
            ]
        }
         """},
    ]
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples
    )

    responseSchema = [
        ResponseSchema(name="date", description="該紀念日的日期", type="YYYY-MM-DD"),
        ResponseSchema(name="name", description="該紀念日的名稱")
    ]
    output_parser = StructuredOutputParser(response_schemas=responseSchema)
    format_instructions = output_parser.get_format_instructions()
    format_instructions = """
    The output should be a markdown code snippet formatted in the following schema, without the leading and trailing "```json" and "```":
    {
        "Result": [
            {
                "date": YYYY-MM-DD  // 該紀念日的日期
                "name": string  // 該紀念日的名稱
            }
        ]
    }
    """
    print("************************** format_instructions ****************************")
    print(format_instructions)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "使用台灣語言並回答問題,{format_instructions1}"),
        few_shot_prompt,
        ("human", "{question1}")
    ])
    prompt = prompt.partial(format_instructions1=format_instructions)
    prompt_str = prompt.format(question1=question)
    print("************************** prompt_str ****************************")
    print(prompt_str)
    return prompt_str

def generate_hw01(question):
    # promptTemplate = PromptTemplate(template="告訴我多個關於{topic}的知識", input_variables=["topic"])
    # question = promptTemplate.format(topic=question)
    # chat_template = ChatPromptTemplate.from_messages(
    #     [
    #         ("system", "使用中文回答問題"),
    #         ("human", "{question}")
    #     ]
    # )
    # prompt_str = chat_template.format(question=question)
    # print(prompt_str)
    # print("******************************************************")

    prompt_str = question_to_prompt(question)
    return demo(prompt_str).content
    
def generate_hw02(question):
    tool = StructuredTool.from_function(
        name="get_calendarific",
        description="get holidays by calendarific API key",
        func=get_calendarific_data,
        args_schema=GetCalendarific
    )
    prompt = hub.pull("hwchase17/openai-functions-agent")
    print("************************** hub.pull ****************************")
    print(prompt.messages)
    tools = [tool]
    agent = create_openai_functions_agent(get_llm(), tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    prompt_str = question_to_prompt(question)
    response = agent_executor.invoke({"input": prompt_str}).get('output')
    print("************************** response ****************************")
    return response
    
def generate_hw03(question2, question3):
    pass
    
def generate_hw04(question):
    pass
    
def demo(question):
    llm = get_llm()
    message = HumanMessage(
            content=[
                {"type": "text", "text": question},
            ]
    )
    print("************************** message ****************************")
    print(message)
    response = llm.invoke([message])

    # print("************************** question ****************************")
    # print(question)
    # response = llm.invoke(question)
    
    print("************************** response ****************************")
    return response

# print(generate_hw01("2024年台灣10月紀念日有哪些?"))
print(generate_hw02("2024年台灣10月紀念日有哪些?"))
# print(generate_hw01("2025年台灣10月紀念日"))