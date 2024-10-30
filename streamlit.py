## here i will be writting code for web application for my question answering chatbot
import streamlit as st
from QAWITHPDF.data_ingestion import load_data
from QAWITHPDF.embedding import download_gemini_embedding
from QAWITHPDF.model_api import load_model
import traceback

def main():
    # Set page configuration
    st.set_page_config(page_title="QA with Documents")

    st.header("QA with Documents (Information Retrieval)")

    # File uploader
    doc = st.file_uploader("Upload your document", type=["pdf", "txt", "docx"])
    user_question = st.text_input("Ask your question")

    if st.button("Submit and Process"):
        if doc is None:
            st.warning("Please upload a document.")
            return

        with st.spinner("Processing..."):
            try:
                # Load document
                document = load_data(doc)
                if not document:
                    st.error("Failed to load the document.")
                    return
                
                # Load model
                model = load_model()
                if not model:
                    st.error("Failed to load the model.")
                    return
                
                # Download embeddings and create query engine
                query_engine = download_gemini_embedding(model, document)
                if not query_engine:
                    st.error("Failed to initialize the query engine.")
                    return

                # Query engine response
                response = query_engine.query(user_question)
                if response:
                    st.write(response.response)
                else:
                    st.error("No response returned from the query engine.")

            except Exception as e:
                st.error("An error occurred during processing.")
                traceback.print_exception(type(e), e, e.__traceback__)

if __name__ == "__main__":
    main()


