import streamlit as st
from dotenv import load_dotenv

import os

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser import TokenTextSplitter

import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext, VectorStoreIndex

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate

from llama_parse import LlamaParse
import getpass

from llama_index.core import SimpleDirectoryReader

from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
import time



qdrant_host = os.getenv('QDRANT_HOST')

# Connect to qdrant client
qdrant_client = qdrant_client.QdrantClient(
    url=qdrant_host
)

def main():
    # Load env variables
    load_dotenv()

    # Configuring llama index
    # Define the vector store
    vector_store = QdrantVectorStore(client=qdrant_client, collection_name='llamaparse 3-pdf files')

    # Create an vector store index
    index = VectorStoreIndex.from_vector_store(vector_store = vector_store)

    st.set_page_config(page_title='LlamaIndex Pipeline')

    st.header('Ask Crypto Regulatory Compliance Question!')
    st.write('Llamaindex, llamaparse and Reranker pipeline')

    # # Grab the user question
    user_question = st.text_input("Ask me any crypto regulatory question!")

    if user_question:

        reranker = FlagEmbeddingReranker(
        top_n=3,
        model="BAAI/bge-reranker-large",
        )

        query_engine = index.as_query_engine(
            similarity_top_k = 5,
            node_postprocessors=[reranker],
            verbose=True,
        )

        

        response = query_engine.query(user_question)

        st.write(f'Question: {user_question}')

        st.markdown(':green[Response:]')
        st.write(response.response)

        sources = response.source_nodes        

        for source in sources:
            st.write(source.text)
            # st.divider()
        #     time.sleep(1)

    

if __name__ == '__main__':
    main()
