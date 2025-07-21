import requests
from typing import Literal, Optional
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")

class ValidatorResponse(BaseModel):
    decision: Literal["summarize", "general"] = Field(description="The decision made by the llm initially.")

class GraphState(BaseModel):
    user_input: str = Field(description="The user's input to the graph.")
    output: Optional[str] = Field(default=None, description="The output of the graph.")
    email_content: Optional[str] = Field(default=None, description="The content of the email to be summarized.")
    decision: Optional[Literal["summarize", "general"]] = Field(default=None, description="The decision made by the llm initially.")

def initial_validator(state: GraphState):
    user_input = state.user_input
    prompt = PromptTemplate(
        template=
        """
            You are a helfpul AI assistant that helps to decide what action based on the user input. 
            Your task is to decide what action to take based on the user input.
            The user input is: {user_input}
            Your decision should be one of the following: summarize, general
        """,
        input_variables=["user_input"]
    )
    llm_with_structured_output = llm.with_structured_output(ValidatorResponse)
    chain = prompt | llm_with_structured_output
    response  = chain.invoke({"user_input": user_input})
    return {"decision": response.decision}

def email_content_validator(state: GraphState):
    user_input = state.user_input
    prompt = PromptTemplate(
        template=
        """
            You are a helpful AI assistant that helps in fetching the email content.
            Your task is to fetch the email content based on the user input.
            The user input is: {user_input}
            Your response should be the email content.
        """,
        input_variables=["user_input"]
    )
    chain = prompt | llm
    response  = chain.invoke({"user_input": user_input})
    return {"email_content": response.content}

def summarize(state: GraphState):
    email_content = state.email_content
    prompt = PromptTemplate(
        template=
        """
            You are a helpful AI assistant that helps in summarizing the email content.
            Your task is to summarize the email content.
            The email content is: {email_content}
            Your response should be a summary of the email content.
        """,
        input_variables=["email_content"]
    )
    chain = prompt | llm
    response  = chain.invoke({"email_content": email_content})
    return {"output": response.content}

def general(_: GraphState):
    return {"output": "This is a general response."}

def router(state: GraphState):
    decision = state.decision
    if decision == "summarize":
        return "summarize"
    elif decision == "general":
        return "general"
    
def send_to_pipedream(state: GraphState):
    summary = state.output
    url = "https://eo8worvi6zb7tlg.m.pipedream.net"
    payload = {"summary": summary}
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
       print("Error sending to Pipedream:", e)
    return state

workflow = StateGraph(GraphState)

workflow.add_node("initial_validator", initial_validator)
workflow.add_node("email_content_validator", email_content_validator)
workflow.add_node("send_to_pipedream", send_to_pipedream)
workflow.add_node("summarize", summarize)
workflow.add_node("general", general)

workflow.add_edge(START, "initial_validator")
workflow.add_conditional_edges("initial_validator", router, {
    "summarize": "email_content_validator",
    "general": "general"
})
workflow.add_edge("email_content_validator", "summarize")
workflow.add_edge("summarize", "send_to_pipedream")
workflow.add_edge("send_to_pipedream", END)
workflow.add_edge("general", END)

graph = workflow.compile()