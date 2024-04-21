## What is Retrieval Augmented Generation
RAG enhances the existing potent functionalities of LLMs for particular domains or an organization's internal knowledge repository, 
without necessitating model retraining. This method offers a cost-efficient means of enhancing LLM output, ensuring its continued relevance, precision, 
and utility across diverse settings. <br>

LLMs are used to generate AI powered intelligent chatbots, computer vision, robotics and consumer devices, etc.
Unfortunately, the nature of LLM technology introduces unpredictability in LLM responses as its not a state machine and it undergoes the following challenges:
1. Hallucination - the phenomena where the LLM responds with false information or partially false information which is problem hard to resolve.
2. Presenting out-of-date or generic information when the user expects a specific, current response.
3. Generating responses from non-authoritative sources and finetuning or training an LLM for a domain specific conversation incurs additional costs.

RAG brings in the following benefits over an LLM:
1. With the help of RAG, LLMs can provide latest information to users by feeding in latest research, statistics, or news/social media feeds to the generative models. This can be 
2. RAG is a cost efficient option and makes an LLM broadly accessible and usable.
3. Both developers and users can not trust LLMs to present accurate information with source attribution increasing trust and confidence in generative AI solution.


## Working of a Retrieval Augmented Generation
Shown below is a sequence diagram with explanation by NVIDIA at a high level.


![image](https://github.com/ShrishtiKarkera/MistralAI-multipdf-RAGBOT/assets/57498417/0056fc9e-8f8c-45b4-9567-92297ad06a18)

- When users ask an LLM a question, the AI model sends the query to another model that converts it into a numeric format so machines can read it. The numeric version of the query is sometimes called an embedding or a vector. <br>
- The embedding model then compares these numeric values to vectors in a machine-readable index of an available knowledge base. <br>
- When it finds a match or multiple matches, it retrieves the related data, converts it to human-readable words and passes it back to the LLM. <br>
- Finally, the LLM combines the retrieved words and its own response to the query into a final answer it presents to the user, potentially citing sources the embedding model found. <br>

## Let's built a chatbot using retrieval augmented generation

We will use Streamlit for the user interface using which users will upload multiple files of their own and then ask questions to the chatbot related to those files. <br>
Let's start with an upload button for users to upload their files, we'll use streamlit's file_uploader for this and place this in the sidebar to make it look like ChatGPT's UI for user familiarity
```python
def main():
    # Initialize session state
    initialize_session_state()
    st.title("Chat with your files using Mistral-7B-Instruct :mag:")
    # Initialize Streamlit
    st.sidebar.title("MultiPDF Document Processing")
    uploaded_files = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)
```
Next, extract text from the uploaded files, you can also use youtube videos, news streams or other sources as the input feed using https://python.langchain.com/docs/integrations/document_loaders/
```python
if uploaded_files:
  text = []
  for file in uploaded_files:
      file_extension = os.path.splitext(file.name)[1]
      with tempfile.NamedTemporaryFile(delete=False) as temp_file:
          temp_file.write(file.read())
          temp_file_path = temp_file.name

      loader = None
      if file_extension == ".pdf":
          loader = PyPDFLoader(temp_file_path)

      if loader:
          text.extend(loader.load())
          os.remove(temp_file_path)
```

The text extracted from multiple pdfs are then divided into chunks using LangChain’s text splitter
```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=20)
text_chunks = text_splitter.split_documents(text)
```
Now, let's convert these chunks into Sentence Transformer embeddings and index it in FAISS vector store (you can use embeddings and vector store of your choice)
```python
# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})

# Create vector store
vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
```
Finally, I used a ConversationalRetrievalChain with Mistral 7B sharded LLM for retrieval. 
```python
# Create the chain object
chain = create_conversational_chain(vector_store)           # in the main function


def create_conversational_chain(vector_store):
    # Create llm
    llm = LlamaCpp(
    streaming = True,
    model_path="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    temperature=0.75,
    top_p=1, 
    verbose=True,
    n_ctx=4096
)
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                 retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                 memory=memory)
    return chain

# Display the final output
display_chat_history(chain)                                # in the main function again
```

All this was run on my CPU, you can also use QLORA-finetuned models, those are light too! <br>

Next, to also display conversational history, follow along the code here - https://github.com/ShrishtiKarkera/MistralAI-multipdf-RAGBOT/tree/main/Chatapp/app.py

I have attached a demonstration below, where I gave first 2 Harry Potter books as pdfs and asked some questions related to that.

[![Watch the video](https://img.youtube.com/vi/PTvkBwq5VnI/default.jpg)](https://youtu.be/PTvkBwq5VnI)


### References:
* https://aws.amazon.com/what-is/retrieval-augmented-generation/
* https://www.linkedin.com/pulse/llms-just-chatbots-akshay-ballal-vzfwe/
* https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/
* https://python.langchain.com/docs/modules/data_connection/document_loaders/
