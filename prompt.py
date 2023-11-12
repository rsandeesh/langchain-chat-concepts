from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from redundant_filter_retriever import RedundantFilterRetriever
from dotenv import load_dotenv
import langchain

langchain.debug = True

load_dotenv()

embeddings = OpenAIEmbeddings()
chat = ChatOpenAI()

# max_marginal_relevance_search_by_vector
db = Chroma(persist_directory="emb", embedding_function=embeddings)

# retriever = RedundantFilterRetriever(
#     embedding=embeddings,
#     chroma=db
# )

retriever = db.as_retriever()

# map_reduce - Build a summary of each document, then feed each summary into a final question
# map_rerank - Find relevant part of each document and give it a score of how relevant it is
# refine = Build an initial response, then give the LLM an opportunity to update it with further context
chain = RetrievalQA.from_chain_type(
    llm=chat, retriever=retriever, chain_type="stuff"  # map_reduce, map_rerank, refine
)

result = chain.run("What is an interesting fact about english language?")

print(result)
