#!/path/to/your/virtualenv/bin/python
import os
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.cli.rag import RagCLI
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
#DOCS https://docs.llamaindex.ai/en/stable/getting_started/starter_tools/rag_cli/
# optional, set any API keys your script may need (perhaps using python-dotenv library instead)
#os.environ["PINECONE_KEY"] = "sk-xxx"

from llama_index.llms.huggingface import HuggingFaceLLM

# Use a smaller, more compatible model
llm = HuggingFaceLLM(
    model_name="google/flan-t5-small",  # Designed for Q&A tasks
    tokenizer_name="google/flan-t5-small",
    max_new_tokens=256,
)

Settings.llm = llm

Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("example_collection")
# Set up the ChromaVectorStore and StorageContext
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
docstore = SimpleDocumentStore()
#embeddings = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

#llm = None
#print(Settings.llm)
custom_ingestion_pipeline = IngestionPipeline(
    #transformations=[...],
    vector_store=vector_store,
    docstore=docstore,
    cache=IngestionCache(),
)



# you can optionally specify your own custom readers to support additional file types.
#file_extractor = {".html": ...}

rag_cli_instance = RagCLI(
    ingestion_pipeline=custom_ingestion_pipeline,
    #file_extractor=file_extractor,  # optional
)

if __name__ == "__main__":
    rag_cli_instance.cli()