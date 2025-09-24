from langchain.prompts import PromptTemplate
class Retrieval:
    def __init__(self, device, index, Embeddings, vector_store):
        self.index = index
        self.device = device
        self.Embeddings = Embeddings
        self.vector_store = vector_store

    def get_retrieval(self):
        template = """Answer the question based only on the following context. 
            If you cannot answer the question with the context, 
            please respond with 'I don't know':

            Context: {context}
            Question: {question}
            """
        prompt = PromptTemplate.from_template(template)
        retriever = self.vector_store.as_retriever()
        return retriever, prompt

# Retriever= Retrieval(device=Device, index=index, 
#                      Embeddings=Embeddings,
#                      vector_store=vector_store)
# retriever = Retriever.get_retrieval()
# docs = retriever.get_relevant_documents("What is Thermodynamics")
# for i, doc in enumerate(docs):
#    print(f'\nResult{i}:\n{doc.page_content}')







