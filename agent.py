
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser  


class CodeOutput(BaseModel):
    code: str = Field(description="The pandas code to execute.")
    # explanation: str = Field(description="The explanation of the code.")


parser = PydanticOutputParser(pydantic_object=CodeOutput)


class AgentState(TypedDict):
    query: str
    df: pd.DataFrame
    code: str
    result: str
    error: str
    retries: int

def generate_code(state: AgentState):
    """Generate pandas code to answer the query."""
    retries = state.get("retries", 0) + 1
    prompt_template = """You are a helpful assistant that generates pandas code to answer a natural language query.

**Query:**
{query}

**Dataframe Head:**
{df_head}

**Instructions:**
1.  Generate pandas code to answer the query.
2.  The code should be a single expression that can be evaluated with `eval()`.
3.  The code should not include any imports.
4.  The dataframe is available in a variable named `df`.

Respond ONLY with a JSON object in the following format:
{format_instructions}
{error_context}
"""
    
    error_context = ""
    if state.get("error"):
        # If there's a previous error, include it in the prompt.
        error_context = f"""
**Previous Attempt Failed:**
**Code:**
```python
{state.get('code', 'N/A')}
```
**Error:**
{state['error']}

Please correct the code.
"""
    
    llm = Ollama(model="llama2") # Replace with your chosen Ollama model
    pydantic_parser = PydanticOutputParser(pydantic_object=CodeOutput)
    output_fixing_parser = OutputFixingParser.from_llm(parser=pydantic_parser, llm=llm)

    prompt = ChatPromptTemplate.from_template(
        prompt_template,
        partial_variables={"format_instructions": pydantic_parser.get_format_instructions()}
    )
    
    chain = prompt | llm | output_fixing_parser
    df_head = state["df"].head().to_string()
    
    try:
        result = chain.invoke({
            "query": state["query"], 
            "df_head": df_head,
            "error_context": error_context
        })
        # On success, return the code, retries, and clear any previous error.
        return {"code": result.code, "retries": retries, "error": None}
    except Exception as e:
        # On generation failure, return the error, retries, and no code.
        return {"error": str(e), "retries": retries, "code": None}

def execute_code(state: AgentState):
    """Execute the generated pandas code."""
    if state.get("code") is None:
        # If no code was generated due to an error, do nothing.
        # The error from the generation step will be passed through.
        return {}
    try:
        # Execute the code
        result = eval(state["code"], {"df": state["df"]})
        # If successful, clear the error and set the result
        return {"result": str(result), "error": None}
    except Exception as e:
        # If there's an execution error, return the error message
        return {"error": str(e)}

def should_continue(state: AgentState):
    """Determine whether to continue or end the process."""
    if state.get("error") and state.get("retries", 0) < 2:
        return "generate_code"
    else:
        return END

# Create the graph
workflow = StateGraph(AgentState)

# Add the nodes
workflow.add_node("generate_code", generate_code)
workflow.add_node("execute_code", execute_code)

# Add the edges
workflow.set_entry_point("generate_code")
workflow.add_edge("generate_code", "execute_code")
workflow.add_conditional_edges(
    "execute_code",
    should_continue,
)

# Compile the graph
app = workflow.compile()
