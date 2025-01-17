import json
import traceback

from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import (ResponseSchema, StructuredOutputParser)

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

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
        ("human", "{question1}")
    ])
    prompt = prompt.partial(format_instructions1=format_instructions)
    prompt_str = prompt.format(question1=question)
    print("************************** prompt_str ****************************")
    print(prompt_str)
    return demo(prompt_str).content
    
def generate_hw02(question):
    pass
    
def generate_hw03(question2, question3):
    pass
    
def generate_hw04(question):
    pass
    
def demo(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
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

print(generate_hw01("2024年台灣10月紀念日有哪些?"))
# print(generate_hw01("2025年台灣10月紀念日"))