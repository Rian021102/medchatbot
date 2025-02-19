from langchain_community.vectorstores import DeepLake
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain



api_key = os.environ.get("OPENAI_API_KEY")
embeddings_model = OpenAIEmbeddings()
ACTIVELOOP_TOKEN = os.getenv("ACTIVELOOP_TOKEN")
llm = ChatOpenAI(model_name = "gpt-4o", max_tokens = 1000)


db=DeepLake(dataset_path="hub://rian/medicaldoc",embedding=embeddings_model,read_only=True)
retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

chain = load_qa_chain(llm, chain_type="stuff")

# A utility function for answer generation
def ask(question):
    # Define the prompt template
    prompt_template = """Answer the following question based only on the provided context.
    Please provide with detail answers. 
    If you don't know the answer, just say you don't know. Don't try to make up an answer.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer: Let's approach this step by step:"""
    
    # Retrieve relevant documents
    context = retriever.get_relevant_documents(question)
    
    # Format the prompt with the context and question
    prompt = prompt_template.format(context=context, question=question)
    
    # Generate the answer using the prompt
    answer = (chain({"input_documents": context, "question": prompt}, return_only_outputs=True))['output_text']
    
    return answer

user_question = "What this document is all about?"

answer = ask(user_question)
print("Answer:", answer)