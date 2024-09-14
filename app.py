from src.helper import download_embedding_model,get_data,get_data_chunks
from langchain_pinecone import PineconeVectorStore
import pinecone
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
from src.prompt import * 
import os
from flask import Flask, render_template, jsonify, request



app=Flask(__name__)


prompt_template="""
Use the following information to answer the users question.
If you don't know the answer or you're not sure about it, say I don't know. don't make up an answer.
Context:{context}
Question:{input}
Only return the answer. 
"""


load_dotenv()
#api_key = os.environ.get('API_KEY')
index_name=os.environ.get('INDEX_NAME')
#data=get_data('data/')
#chunks=get_data_chunks(data)
embedding_model=download_embedding_model()
#docsearch=PineconeVectorStore.from_texts([t.page_content for t in chunks], embedding_model, index_name=index_name)
docsearch=PineconeVectorStore.from_existing_index(index_name, embedding_model)
llm=CTransformers(model="llama-2-7b-chat.ggmlv3.q4_0.bin",model_type="llama")
retriever=docsearch.as_retriever(search_kwargs={'k':1})
prompt=PromptTemplate(template=prompt_template,input_variables=["context","input"])
question_answer_chain = create_stuff_documents_chain(llm, prompt)
chain = create_retrieval_chain(retriever, question_answer_chain)



@app.route("/")
def index():
    return render_template("mchatbot.html")
@app.route("/get",methods=['GET','POST'])
def ask_chat():
    input=request.form["msg"]
    result=chain.invoke({"context":'serious',"input":input})
    print("Response : ", result)
    print('type of result',type(result))
    #return str(result['context'][0].page_content)
    return str(result['answer'])

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)