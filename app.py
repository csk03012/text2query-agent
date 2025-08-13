
import streamlit as st
import pandas as pd
from agent import app
from dotenv import load_dotenv
import os

load_dotenv()

st.title("Text2Query Agent")

# File uploader
uploaded_files = st.file_uploader("Upload your Excel or CSV files", type=["xlsx", "csv"], accept_multiple_files=True)

if uploaded_files:
    dataframes = {}
    for uploaded_file in uploaded_files:
        # Read the file based on its extension
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)

        # Remove unnamed columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # Add the dataframe to the dictionary
        dataframes[uploaded_file.name] = df

    for name, df in dataframes.items():
        st.write(f"**Dataframe Head: {name}**")
        st.write(df.head())

    # Context input
    context = st.text_area("Provide context about the dataframe (optional)")

    # Query input
    query = st.text_input("Enter your natural language query:")

    # Inference options
    inference_option = st.radio("Select Inference Option", ("Ollama", "Hugging Face Inference API"))
    hf_token = ""
    if inference_option == "Hugging Face Inference API":
        hf_token = st.text_input("Enter your Hugging Face Token", type="password", value=os.getenv("HF_TOKEN"))

    if st.button("Run Query"):
        with st.spinner("Agent is thinking..."):
            st.write("---")
            st.subheader("Agent Execution Log")
            last_code = ""
            
            # Stream the agent's execution
            for i, event in enumerate(app.stream({"query": query, "df": dataframes, "context": context, "inference_option": inference_option, "hf_token": hf_token})):
                for node_name, node_output in event.items():
                    if node_name == "__end__":
                        st.subheader("Final Result")
                        if node_output.get("error"):
                            st.error(f"Agent stopped after {node_output.get('retries', 'N/A')} attempts.")
                            st.error(f"Final Error: {node_output.get('error')}")
                        else:
                            st.success("Query executed successfully!")
                            st.write(node_output.get("result"))
                        st.write("---")
                        continue

                    st.markdown(f"### Step {i+1}: Running `{node_name}`")
                    
                    if node_name == "analyze_dataframe":
                        st.markdown("**DataFrame Analysis:**")
                        st.write(node_output["dataframe_analysis"])
                    elif node_name == "enhance_query":
                        st.markdown("**Enhanced Query:**")
                        st.write(node_output["enhanced_query"])
                    elif node_name == "generate_code":
                        if node_output.get("code"):
                            last_code = node_output["code"]
                            st.markdown("**Generated Code:**")
                            st.code(last_code, language="python")
                        if node_output.get("error"):
                            st.markdown("**Generation Error:**")
                            st.warning(node_output["error"])

                    elif node_name == "execute_code":
                        st.markdown("**Executed Code:**")
                        st.code(last_code, language="python")
                        if node_output.get("result"):
                            st.markdown("**Execution Result:**")
                            st.write(node_output["result"])
                        if node_output.get("error"):
                            st.markdown("**Execution Error:**")
                            st.warning(node_output["error"])
                    elif node_name == "verify_code":
                        if node_output.get("verification_feedback"):
                            st.markdown("**Verification Feedback:**")
                            st.warning(node_output["verification_feedback"])
                st.write("---")
