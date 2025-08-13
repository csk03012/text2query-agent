import json
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Dict
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser


def get_llm(inference_option, hf_token):
    if inference_option == "Ollama":
        return Ollama(model="qwen3:8b")
    else:
        hf_llm = ChatOpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=hf_token,
            model="Qwen/Qwen3-8B:nscale",
        )
        return hf_llm

class CodeOutput(BaseModel):
    code: str = Field(description="A multi-line block of Python code to execute.")

parser = PydanticOutputParser(pydantic_object=CodeOutput)


class AgentState(TypedDict):
    query: str
    df: Dict[str, pd.DataFrame]
    code: str
    result: str
    error: str
    retries: int
    context: str
    inference_option: str
    hf_token: str
    dataframe_analysis: str
    enhanced_query: str
    verification_feedback: str

def analyze_dataframe(state: AgentState):
    """Analyze the dataframe and generate a structured summary."""
    df_summaries = {}
    for name, df in state["df"].items():
        df_summaries[name] = {
            "columns": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "head": df.head().to_string(),
        }

    prompt = f"""You are a data analyst. For each dataframe provided, do the following:

1.  **Summarize Each Column:** For each column, provide a one-sentence summary (10-20 words) of its content and purpose.
2.  **Summarize the Data:** Provide a brief, high-level summary of the data in a few bullet points.

**Dataframe Summaries:**
{df_summaries}

**Analysis:**
"""

    llm = get_llm(state["inference_option"], state["hf_token"])
    if state["inference_option"] == "Ollama":
        response = llm.invoke(prompt)
    else:
        response = llm.invoke(prompt).content

    return {"dataframe_analysis": response}

def enhance_query(state: AgentState):
    """Enhance the user query and create a step-by-step plan."""
    prompt = f"""You are a cricket analyst. Your task is to enhance the user's query and create a clear, step-by-step plan to answer it using the provided dataframe summaries.

**Dataframe Analysis:**
{state["dataframe_analysis"]}

**User Query:**
{state["query"]}

**Instructions:**
1.  **Enhance the Query:** Rewrite the user's query to be more specific and detailed, based on the available columns in the dataframes.
2.  **Create a Step-by-Step Plan:** Outline the sequence of pandas operations required to answer the enhanced query. Be specific about which dataframes to use and which columns to join or filter on.
3. **Very Important: Do not generate Code only Plan.
**Enhanced Query and Plan:**
"""

    llm = get_llm(state["inference_option"], state["hf_token"])
    if state["inference_option"] == "Ollama":
        response = llm.invoke(prompt)
    else:
        response = llm.invoke(prompt).content

    return {"enhanced_query": response}

def generate_code(state: AgentState):
    """Generate pandas code to answer the query."""
    state["error"] = None
    state["code"] = None
    retries = state.get("retries", 0) + 1
    prompt_template = """You are a helpful assistant that generates pandas code to answer a natural language query.

**Query:**
{query}

**Enhanced Query:**
{enhanced_query}

**Dataframe Heads:**
{df_head}

**Context:**
{context}

**Instructions:**
1. Use the dataframes in the `dfs` dictionary as the input. The keys of the dictionary are the filenames.
2. Break the logic into **multiple steps** using meaningful variable names.
3. Save **intermediate results** to variables (e.g., `grouped_df`, `filtered_df`, etc.).
4. Do not include imports.
5. Strictly include the code for preprocessing or cleaning step before answering the asked question.
6. **The final answer must be stored in a variable named `result`**.
7. Return **only valid Python code**.

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

**Verification Feedback:**
{state.get('verification_feedback', 'N/A')}

Please correct the code.
"""
    
    llm = get_llm(state["inference_option"], state["hf_token"])
    pydantic_parser = PydanticOutputParser(pydantic_object=CodeOutput)
    output_fixing_parser = OutputFixingParser.from_llm(parser=pydantic_parser, llm=llm)

    prompt = ChatPromptTemplate.from_template(
        prompt_template,
        partial_variables={"format_instructions": pydantic_parser.get_format_instructions()}
    )
    
    chain = prompt | llm
    df_heads = {name: df.head().to_string() for name, df in state["df"].items()}
    
    try:
        # Get the raw output from the LLM
        raw_output = chain.invoke({
            "query": state["query"],
            "enhanced_query": state["enhanced_query"],
            "df_head": df_heads,
            "context": state["context"],
            "error_context": error_context
        })
        
        # Extract the content from the message if using HF
        if state["inference_option"] == "Hugging Face Inference API":
            raw_output = raw_output.content

        # Extract the JSON part of the output
        try:
            # Find the start and end of the JSON object
            start_index = raw_output.find('{')
            end_index = raw_output.rfind('}') + 1
            json_string = raw_output[start_index:end_index]
            
            # Parse the extracted JSON
            parsed_json = json.loads(json_string)
            result = CodeOutput(**parsed_json)
            
            # On success, return the code, retries, and clear any previous error.
            return {"code": result.code, "retries": retries, "error": None}
        except (json.JSONDecodeError, IndexError) as e:
            # If JSON extraction or parsing fails, try to fix it with the parser
            result = output_fixing_parser.parse(raw_output)
            return {"code": result.code, "retries": retries, "error": None}
    except Exception as e:
        # On generation failure, return the error, retries, and no code.
        return {"error": str(e), "retries": retries, "code": {}}

def verify_code(state: AgentState):
    """Verify the generated code against the query and dataframe analysis."""
    if state.get("code") is None:
        return {"verification_feedback": "No code to verify."}

    prompt = f"""You are a code verifier. Your task is to verify if the given Python code correctly answers the query based on the dataframe analysis and enhanced query.

**Query:**
{state["query"]}

**Enhanced Query:**
{state["enhanced_query"]}

**Dataframe Analysis:**
{state["dataframe_analysis"]}

**Generated Code:**
```python
{state["code"]}
```

Is the code correct? If not, provide feedback on what is wrong.
Start your response with "Correct" or "Incorrect". If incorrect, provide a reason.
"""
    llm = get_llm(state["inference_option"], state["hf_token"])
    if state["inference_option"] == "Ollama":
        response = llm.invoke(prompt)
    else:
        response = llm.invoke(prompt).content

    if response.strip().startswith("Correct"):
        return {"verification_feedback": None}
    else:
        return {"verification_feedback": response}

def execute_code(state: AgentState):
    """Execute the generated pandas code."""
    if state.get("code") is None:
        return {}
    
    local_vars = {"dfs": state["df"], "pd": pd}
    try:
        # Execute the multi-line code block
        exec(state["code"], local_vars)
        # The final answer is in the 'result' variable
        result = local_vars.get("result", "No result variable found.")
        return {"result": str(result), "error": None}
    except Exception as e:
        # If there's an execution error, return the error message
        return {"error": str(e)}

def should_continue(state: AgentState):
    """Determine whether to continue or end the process."""
    if state.get("error") and state.get("retries", 0) < 3:
        return "generate_code"
    else:
        return END

def should_regenerate(state: AgentState):
    """Determine whether to regenerate code based on verification feedback."""
    if state.get("verification_feedback") and state.get("retries", 0) < 2:
        return "generate_code"
    else:
        return "execute_code"

# Create the graph
workflow = StateGraph(AgentState)

# Add the nodes
workflow.add_node("analyze_dataframe", analyze_dataframe)
workflow.add_node("enhance_query", enhance_query)
workflow.add_node("generate_code", generate_code)
workflow.add_node("verify_code", verify_code)
workflow.add_node("execute_code", execute_code)

# Add the edges
workflow.set_entry_point("analyze_dataframe")
workflow.add_edge("analyze_dataframe", "enhance_query")
workflow.add_edge("enhance_query", "generate_code")
workflow.add_edge("generate_code", "verify_code")
workflow.add_conditional_edges(
    "verify_code",
    should_regenerate,
)
workflow.add_conditional_edges(
    "execute_code",
    should_continue,
)

# Compile the graph
app = workflow.compile()