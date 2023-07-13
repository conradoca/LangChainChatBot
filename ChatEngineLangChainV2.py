import os
from dotenv import load_dotenv, find_dotenv
from langchain.agents import Tool, tool, AgentOutputParser
from langchain.agents import ZeroShotAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.schema import AgentAction, AgentFinish
from typing import List, Union
import urllib3
import re
import json
from commonUtils import configParams

class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(
            tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output
        )



class LanguageModel:

    def __init__(self, temperature=0.0, model="gpt-3.5-turbo"):
        _ = load_dotenv(find_dotenv()) # read local .env file
        config = configParams()
        params = config.loadConfig()

        self.temperature = temperature
        self.model = model
        llm = ChatOpenAI(
                    temperature=self.temperature, 
                    model=self.model,
                    verbose=False
                    )
        embeddings = OpenAIEmbeddings()
        # Creating a QA retriever to search on the vector store
        productInfoDB = FAISS.load_local(params['VectorStore'], embeddings)

        productInfo = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff", 
            retriever=productInfoDB.as_retriever()
            )
        
        tools = [
            self.orderStatus,
            Tool(
                name="Product Information QA System",
                func=productInfo.run,
                description="useful for when you need to answer questions about product information. Input should be a fully formed question about a product offered by Pixartprinting.",
            ),
        ]

        prefix = """You are an customer care agent for a company called Pixart. You will receive questions from Pixart's customers. You will always response as if you are Pixartprinting.
            The topics that you will be answering are: 
            1) Product information: Provide information about the products sold by Pixart, the product types, materials and usage of these products 
            2) Order Status: The customer should provide an order number and you will respond back informing about the order's status and the items that make that order.
                if the order number is not present on the question or on the wrong format please ask the customer to enter the correct number. The order number 
                    follows the format of this regular expression: r"(?<![A-Za-z0-9])([A-Z]{3}-[A-Z]{3}-\d{6})(?![A-Za-z0-9])" 
            To answer these two types of questions you have access to the following tools: 
            {tools}
            
            The answer should be as extended and detailed as possible.
            If you receive any questions that you think you can't answer ask the customer to call Pixart's customer care by telephone.
            If you receive messages not related to the topics above respond on a kind way that you are answering it.
            """
        suffix = """Begin!"

            {chat_history}
            Question: {input}
            {agent_scratchpad}"""

        prefix = """Have a conversation with a human, answering the following questions as best you can. You have access to the following tools:"""

        prompt = ZeroShotAgent.create_prompt(
                    tools,
                    prefix=prefix,
                    suffix=suffix,
                    input_variables=["input", "chat_history", "agent_scratchpad"]
                    )
        memory = ConversationBufferMemory(memory_key="chat_history")

        output_parser = CustomOutputParser()

        llm_chain = LLMChain(llm=llm, prompt=prompt)
        #agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, output_parser=output_parser, verbose=True)
        agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
        self.agent_chain = AgentExecutor.from_agent_and_tools(
                agent=agent, tools=tools, verbose=True, memory=memory
                )
    
    @tool
    def orderStatus(orderNumber: str) -> tuple[str, dict]:
        """Returns the status of an order number and the items which are part of the order. Use this for any \
        questions related to knowing whether an order is payed, shipped, in production or waiting for prepress. \
        The input should always be a string with the order number. This function will first validate that this is a valid order number. \
        If the order number is a correct order number and it is identified on our systems it will response with a tuple. The elements in this tuple \
        will be: \
        Status: Current Status of that order number
        orderItems: it provides a list of dictionaries. That dictionary contains two keys: item number and the description of that item.\
        """
        config = configParams()
        params = config.loadConfig()
        headers_data = {"Content-Type": "application/json", "Authorization": f"Bearer {params['token']}"}
        http = urllib3.PoolManager()
        orderItems = []

        # Checks the order number format
        #pattern = r"^(PIX|EFL|GIF)-[A-Z]{3}-\d{6}$"
        pattern = r"(?<![A-Za-z0-9])([A-Za-z0-9]{3}-[A-Za-z0-9]{3}-\d{6})(?![A-Za-z0-9])"
        match = re.search(pattern, orderNumber)
        if match is None:
            Status = "Order number not recognised. The order number is not correct"
        else:
            orderNumber = match.group()
            Status = ""
            
            try:
                response = http.request('GET', f"https://order.india.pixartprinting.net/api/v2/order/{orderNumber}", headers=headers_data)
                json_data = json.loads(response.data)

                Status = json_data["data"]["state"]
                itemsList = json_data["data"].get("lineItems")
                for itemID, itemDetails in itemsList.items():
                    itemNumber = itemDetails["lineItemNumber"]
                    itemDescription = itemDetails["merchantInfo"]["merchantProductName"]
                    orderItems.append([itemNumber, itemDescription])
                    
            
            except Exception as e:
                Status = f"We couldn't retrieve the status of the order {orderNumber}. Error: {e}"
            
        return Status, orderItems
   
    def setModel(self, model):
        self.model=model
    
    def response(self, messages):
        with get_openai_callback() as cb:
            completion = {}
            #response=self.productInfo({"query": messages})
            try:
                response=self.agent_chain.run(input=f"{messages}")
                completion["Answer"] = response
                completion["Total Tokens"] = cb.total_tokens
                completion["Prompt Tokens"] = cb.prompt_tokens
                completion["Completion Tokens"] = cb.completion_tokens
                completion["Total Cost USD"] = cb.total_cost
            except Exception as e:
                completion["Answer"] = "I had problems retrieving the information for you."
                completion["Total Tokens"] = cb.total_tokens
                completion["Prompt Tokens"] = cb.prompt_tokens
                completion["Completion Tokens"] = cb.completion_tokens
                completion["Total Cost USD"] = cb.total_cost
        return completion