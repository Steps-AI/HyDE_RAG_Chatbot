import os
from dotenv import load_dotenv,find_dotenv
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.settings import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser
import weaviate
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.postprocessor import SentenceTransformerRerank
os.environ["OPENAI_API_KEY"] = "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"
Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
Settings.embed_model = OpenAIEmbedding()


# Load data
documents = SimpleDirectoryReader(
        input_files=["./data/paul_graham_essay.txt"]
).load_data()

node_parser = SimpleNodeParser.from_defaults(chunk_size=1024)

# Extract nodes from documents
nodes = node_parser.get_nodes_from_documents(documents)

# Connect to your Weaviate instance
client = weaviate.Client(
    embedded_options=weaviate.embedded.EmbeddedOptions(), 
)

index_name = "MyExternalContext"

# Construct vector store
vector_store = WeaviateVectorStore(
    weaviate_client = client, 
    index_name = index_name
)

# Set up the storage for the embeddings
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Setup the index
# build VectorStoreIndex that takes care of chunking documents
# and encoding chunks to embeddings for future retrieval
index = VectorStoreIndex(
    nodes,
    storage_context = storage_context,
)

query_engine = index.as_query_engine()

response = query_engine.query(
    "What happened at Interleaf?"
)