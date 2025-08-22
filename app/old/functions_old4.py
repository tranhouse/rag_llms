from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from typing import Dict, Any

import os
import tempfile
import uuid
import pandas as pd
import re
import warnings
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
import io

# Suppress PyPDF warnings about malformed PDFs
warnings.filterwarnings("ignore", message="Ignoring wrong pointing object")

def clean_filename(filename):
    """
    Remove "(number)" pattern from a filename 
    (because this could cause error when used as collection name when creating Chroma database).

    Parameters:
        filename (str): The filename to clean

    Returns:
        str: The cleaned filename
    """
    # Regular expression to find "(number)" pattern
    new_filename = re.sub(r'\s\(\d+\)', '', filename)
    return new_filename

def process_document_with_ocr(uploaded_file, doc_intelligence_endpoint, doc_intelligence_key):
    """
    Process a PDF document using Azure Document Intelligence OCR API.
    
    Parameters:
        uploaded_file (file-like object): The uploaded PDF file
        doc_intelligence_endpoint (str): Azure Document Intelligence endpoint URL
        doc_intelligence_key (str): Azure Document Intelligence API key
        
    Returns:
        list: A list of Document objects with OCR-extracted text
    """
    try:
        # Initialize Document Intelligence client
        credential = AzureKeyCredential(doc_intelligence_key)
        doc_analysis_client = DocumentAnalysisClient(
            endpoint=doc_intelligence_endpoint, 
            credential=credential
        )
        
        # Read the uploaded file
        file_content = uploaded_file.read()
        
        # Reset file pointer for potential future use
        uploaded_file.seek(0)
        
        # Create a file-like object from bytes
        file_stream = io.BytesIO(file_content)
        
        # Analyze document with OCR
        poller = doc_analysis_client.begin_analyze_document(
            "prebuilt-layout",  # Use layout model for comprehensive text extraction
            document=file_stream
        )
        
        # Get the analysis result
        result = poller.result()
        
        # Extract text content from all pages
        documents = []
        
        if result.pages:
            for page_num, page in enumerate(result.pages):
                page_text = ""
                
                # Extract text from paragraphs (better structure preservation)
                if result.paragraphs:
                    page_paragraphs = [p for p in result.paragraphs 
                                     if any(span.page_number == page_num + 1 
                                           for span in p.spans)]
                    
                    for paragraph in page_paragraphs:
                        page_text += paragraph.content + "\n\n"
                
                # Fallback: extract from lines if no paragraphs
                if not page_text.strip() and page.lines:
                    for line in page.lines:
                        page_text += line.content + "\n"
                
                # Create Document object for each page
                if page_text.strip():
                    doc = Document(
                        page_content=page_text.strip(),
                        metadata={
                            "source": uploaded_file.name,
                            "page": page_num,
                            "extraction_method": "azure_document_intelligence",
                            "page_width": page.width if hasattr(page, 'width') else None,
                            "page_height": page.height if hasattr(page, 'height') else None,
                            "page_unit": page.unit if hasattr(page, 'unit') else None
                        }
                    )
                    documents.append(doc)
        
        # If no pages found, try to extract all text as single document
        if not documents and result.content:
            doc = Document(
                page_content=result.content,
                metadata={
                    "source": uploaded_file.name,
                    "page": 0,
                    "extraction_method": "azure_document_intelligence_fallback"
                }
            )
            documents.append(doc)
            
        return documents
        
    except Exception as e:
        # If OCR fails, provide detailed error
        raise Exception(f"Document Intelligence OCR failed: {str(e)}. Please check your endpoint and API key.")


def get_pdf_text_with_fallback(uploaded_file, doc_intelligence_endpoint=None, doc_intelligence_key=None):
    """
    Process PDF with Document Intelligence OCR first, fallback to PyPDF if needed.
    
    Parameters:
        uploaded_file (file-like object): The uploaded PDF file
        doc_intelligence_endpoint (str): Azure Document Intelligence endpoint URL (optional)
        doc_intelligence_key (str): Azure Document Intelligence API key (optional)
        
    Returns:
        list: A list of Document objects
    """
    # Try Document Intelligence OCR first if credentials provided
    if doc_intelligence_endpoint and doc_intelligence_key:
        try:
            return process_document_with_ocr(uploaded_file, doc_intelligence_endpoint, doc_intelligence_key)
        except Exception as ocr_error:
            print(f"OCR processing failed: {ocr_error}")
            print("Falling back to traditional PDF parsing...")
    
    # Fallback to traditional PDF parsing
    return get_pdf_text_traditional(uploaded_file)


def get_pdf_text_traditional(uploaded_file): 
    """
    Load a PDF document using traditional PyPDF method (fallback).
    
    Parameters:
        uploaded_file (file-like object): The uploaded PDF file to load

    Returns:
        list: A list of documents created from the uploaded PDF file
    """
    try:
        # Read file content
        input_file = uploaded_file.read()

        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(input_file)
        temp_file.close()

        # Suppress PDF parsing warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Ignoring wrong pointing object")
            
            # load PDF document
            loader = PyPDFLoader(temp_file.name)
            documents = loader.load()

        return documents
    
    except Exception as e:
        # If PDF parsing fails completely, raise a more informative error
        raise Exception(f"Failed to parse PDF: {str(e)}. The PDF might be corrupted or password-protected.")
    
    finally:
        # Ensure the temporary file is deleted when we're done with it
        if 'temp_file' in locals():
            os.unlink(temp_file.name)


def split_document(documents, chunk_size, chunk_overlap):    
    """
    Function to split generic text into smaller chunks with improved settings.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],  # Added more separators
        keep_separator=True  # Keep separators to maintain context
    )
    
    return text_splitter.split_documents(documents)


def get_embedding_function(api_key):
    """
    Return an OpenAIEmbeddings object, which is used to create vector embeddings from text.
    The embeddings model used is "text-embedding-ada-002" and the OpenAI API key is provided
    as an argument to the function.

    Parameters:
        api_key (str): The OpenAI API key to use when calling the OpenAI Embeddings API.

    Returns:
        OpenAIEmbeddings: An OpenAIEmbeddings object, which can be used to create vector embeddings from text.
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002", openai_api_key=api_key
    )
    return embeddings


def create_vectorstore(chunks, embedding_function, file_name, vector_store_path="db"):

    """
    Create a vector store from a list of text chunks.

    :param chunks: A list of generic text chunks
    :param embedding_function: A function that takes a string and returns a vector
    :param file_name: The name of the file to associate with the vector store
    :param vector_store_path: The directory to store the vector store

    :return: A Chroma vector store object
    """

    # Create a list of unique ids for each document based on the content
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in chunks]
    
    # Ensure that only unique docs with unique ids are kept
    unique_ids = set()
    unique_chunks = []
    
    unique_chunks = [] 
    for chunk, id in zip(chunks, ids):     
        if id not in unique_ids:       
            unique_ids.add(id)
            unique_chunks.append(chunk)        

    # Create a new Chroma database from the documents
    vectorstore = Chroma.from_documents(documents=unique_chunks, 
                                        collection_name=clean_filename(file_name),
                                        embedding=embedding_function, 
                                        ids=list(unique_ids), 
                                        persist_directory = vector_store_path)

    # Note: Chroma 0.4.x+ automatically persists data, no manual persist() needed
    
    return vectorstore


def create_vectorstore_from_texts(documents, api_key, file_name, doc_intelligence_endpoint=None, doc_intelligence_key=None):
    """
    Create a vector store from a list of texts with OCR processing.
    """
    # Process documents with OCR first
    if doc_intelligence_endpoint and doc_intelligence_key:
        print("Processing document with Azure Document Intelligence OCR...")
        processed_docs = documents  # Documents already processed by OCR in the calling function
    else:
        print("Using traditional PDF parsing...")
        processed_docs = documents
    
    # Improved chunking settings for better context preservation
    docs = split_document(processed_docs, chunk_size=1500, chunk_overlap=300)
    
    # Define embedding function
    embedding_function = get_embedding_function(api_key)

    # Create a vector store  
    vectorstore = create_vectorstore(docs, embedding_function, file_name)
    
    return vectorstore


def load_vectorstore(file_name, api_key, vectorstore_path="db"):

    """
    Load a previously saved Chroma vector store from disk.

    :param file_name: The name of the file to load (without the path)
    :param api_key: The OpenAI API key used to create the vector store
    :param vectorstore_path: The path to the directory where the vector store was saved (default: "db")
    
    :return: A Chroma vector store object
    """
    embedding_function = get_embedding_function(api_key)
    return Chroma(persist_directory=vectorstore_path, 
                  embedding_function=embedding_function, 
                  collection_name=clean_filename(file_name))

# Enhanced prompt template with better instructions
PROMPT_TEMPLATE = """
You are an expert document analysis assistant. Your task is to extract specific information from the provided context.

INSTRUCTIONS:
1. Read through ALL the provided context carefully
2. Extract the requested information accurately
3. If information is not found, clearly state "Information not found in document"
4. Use EXACT text from the document when possible
5. For sources, quote the most relevant sentences directly
6. Be thorough in your reasoning - explain how you found the information

CONTEXT:
{context}

---

EXTRACTION REQUEST: {question}

IMPORTANT: Base your answers ONLY on the provided context. Do not make assumptions or add information not present in the document.
"""

class AnswerWithSources(BaseModel):
    """An answer to the question, with sources and reasoning."""
    answer: str = Field(description="Answer to question")
    sources: str = Field(description="Full direct text chunk from the context used to answer the question")
    reasoning: str = Field(description="Explain the reasoning of the answer based on the sources")


def create_dynamic_model(fields_list):
    """
    Create a dynamic Pydantic model based on user-specified fields.
    
    Parameters:
        fields_list (list): List of field names that the user wants to extract
        
    Returns:
        BaseModel: A dynamically created Pydantic model
    """
    # Create a dictionary to hold the field definitions
    field_definitions = {}
    
    # Add each user-specified field as an AnswerWithSources type
    for field in fields_list:
        # Clean field name to be a valid Python identifier
        clean_field = field.lower().replace(" ", "_").replace("-", "_")
        field_definitions[clean_field] = (AnswerWithSources, Field(description=f"Information about {field}"))
    
    # Create the dynamic model
    DynamicExtractedInfo = type(
        "DynamicExtractedInfo",
        (BaseModel,),
        {
            "__annotations__": {name: field_type for name, (field_type, _) in field_definitions.items()},
            **{name: field_obj for name, (_, field_obj) in field_definitions.items()}
        }
    )
    
    return DynamicExtractedInfo


def format_docs(docs):
    """
    Format a list of Document objects into a single string.

    :param docs: A list of Document objects

    :return: A string containing the text of all the documents joined by two newlines
    """
    return "\n\n".join(doc.page_content for doc in docs)


def query_document(vectorstore, fields_list, api_key, model_name):
    """
    Query a vector store with dynamic fields and return a structured response.
    """
    llm = ChatOpenAI(
        model=model_name, 
        api_key=api_key,
        temperature=0,  # More deterministic results
        max_tokens=4000  # Allow longer responses
    )

    # Enhanced retriever with more results
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 8  # Get more chunks for better context
        }
    )
    
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    
    # Create dynamic model based on user fields
    DynamicModel = create_dynamic_model(fields_list)
    
    # Create a more specific query string
    query = f"""Please extract the following specific information from the document:
    
    Fields to extract: {', '.join(fields_list)}
    
    For each field, provide:
    1. The exact information found
    2. The source text where you found it
    3. Your reasoning for the extraction
    
    If any field cannot be found, clearly state that it's not available in the document."""

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm.with_structured_output(DynamicModel, strict=False)  # Less strict for better results
    )

    try:
        structured_response = rag_chain.invoke(query)
        
        # Convert to dictionary using model_dump instead of dict()
        response_dict = structured_response.model_dump()
        
        # Create DataFrame from the response
        df = pd.DataFrame([response_dict])

        # Transform into a table with three rows: 'answer', 'source', and 'reasoning'
        answer_row = []
        source_row = []
        reasoning_row = []

        for col in df.columns:
            answer_row.append(df[col][0]['answer'])
            source_row.append(df[col][0]['sources'])
            reasoning_row.append(df[col][0]['reasoning'])

        # Create new dataframe with three rows
        structured_response_df = pd.DataFrame(
            [answer_row, source_row, reasoning_row], 
            columns=df.columns, 
            index=['answer', 'source', 'reasoning']
        )
      
        return structured_response_df.T
        
    except Exception as e:
        # If structured output fails, provide error information
        error_df = pd.DataFrame({
            'Error': [f"Extraction failed: {str(e)}", "Please try with a different model or simpler fields", "Check if the document contains the requested information"]
        }, index=['answer', 'source', 'reasoning'])
        return error_df