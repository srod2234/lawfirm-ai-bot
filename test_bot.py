from llama_index.core import VectorStoreIndex, Document, ServiceContext
from llama_index.embeddings.openai import OpenAIEmbedding
from load_pdf import extract_text_from_pdfs  # <-- This calls your new script

# 1. Read and extract all text from PDFs in the 'data/' folder
pdf_text = extract_text_from_pdfs("data")

# 2. Wrap the raw text in a LlamaIndex Document object
documents = [Document(text=pdf_text)]

# 3. Set up OpenAI as your embedding model
embed_model = OpenAIEmbedding()
service_context = ServiceContext.from_defaults(embed_model=embed_model)

# 4. Build an index from the document(s)
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

# 5. Create a query engine that lets you ask questions
query_engine = index.as_query_engine()

# 6. Ask your first question
response = query_engine.query("What is this document about?")
print(response.response)
