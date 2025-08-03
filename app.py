
import streamlit as st
import pandas as pd
from agent import app

st.title("Text2Query Agent")

# File uploader
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("**Dataframe Head:**")
    st.write(df.head())

    # Query input
    query = st.text_input("Enter your natural language query:")

    if st.button("Run Query"):
        with st.spinner("Agent is thinking..."):
            st.write("---")
            st.subheader("Agent Execution Log")
            last_code = ""
            
            # Stream the agent's execution
            for i, event in enumerate(app.stream({"query": query, "df": df})):
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
                    
                    if node_name == "generate_code":
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
                st.write("---")
