import streamlit as st  
from functions import *
import base64

# Initialize session state variables
if 'api_key' not in st.session_state:
    st.session_state.api_key = ''

if 'doc_intelligence_endpoint' not in st.session_state:
    st.session_state.doc_intelligence_endpoint = ''

if 'doc_intelligence_key' not in st.session_state:
    st.session_state.doc_intelligence_key = ''

if 'fields_list' not in st.session_state:
    st.session_state.fields_list = []

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

if 'selected_model' not in st.session_state:
    st.session_state.selected_model = 'gpt-4o-mini'

if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []

def display_pdf(uploaded_file):
    """
    Display a PDF file that has been uploaded to Streamlit.
    Uses a fixed width approach to ensure consistent display across tabs.

    Parameters
    ----------
    uploaded_file : UploadedFile
        The uploaded PDF file to display.

    Returns
    -------
    None
    """
    # Read file as bytes:
    bytes_data = uploaded_file.getvalue()
    
    # Convert to Base64
    base64_pdf = base64.b64encode(bytes_data).decode('utf-8')
    
    # Use a fixed width that works well in Streamlit's column layout
    # This ensures consistent sizing regardless of tab context
    pdf_display = f'''
    <iframe 
        src="data:application/pdf;base64,{base64_pdf}" 
        width="800" 
        height="1000" 
        type="application/pdf"
        style="border: 1px solid #ccc;">
    </iframe>
    '''
    
    # Display file
    st.markdown(pdf_display, unsafe_allow_html=True)


def load_streamlit_page():
    """
    Load the streamlit page with two columns. The left column contains configuration and controls,
    and the right column contains document previews and results.

    Returns:
        col1: The left column Streamlit object.
        col2: The right column Streamlit object.
        uploaded_files: The uploaded PDF files.
    """
    st.set_page_config(layout="wide", page_title="Dynamic PDF Extraction Tool - Multi-Document")

    # Design page layout with 2 columns: File uploader on the left, and other interactions on the right.
    col1, col2 = st.columns([0.5, 0.5], gap="large")

    with col1:
        st.header("Configuration")
        
        # API Key input
        st.subheader("🔑 API Keys")
        st.text_input('OpenAI API Key:', type='password', key='api_key',
                    label_visibility="collapsed", disabled=False,
                    placeholder="Enter your OpenAI API key...")
        
        # Document Intelligence configuration
        st.text_input('Azure Document Intelligence Endpoint:', key='doc_intelligence_endpoint',
                    label_visibility="collapsed", disabled=False,
                    placeholder="https://your-resource.cognitiveservices.azure.com/")
        
        st.text_input('Azure Document Intelligence Key:', type='password', key='doc_intelligence_key',
                    label_visibility="collapsed", disabled=False,
                    placeholder="Enter your Document Intelligence API key...")
        
        # OCR Status indicator
        has_ocr_config = bool(st.session_state.doc_intelligence_endpoint.strip() and st.session_state.doc_intelligence_key.strip())
        if has_ocr_config:
            st.success("✅ OCR Enabled - Documents will be processed with Azure Document Intelligence")
        else:
            st.warning("⚠️ OCR Disabled - Using traditional PDF parsing (may miss text in images/scanned docs)")
        
        # Model selection
        st.subheader("🤖 OpenAI Model Selection")
        model_options = {
            'GPT-5': 'gpt-5',
            'GPT-4o Mini (Recommended)': 'gpt-4o-mini',
            'GPT-4 Turbo': 'gpt-4-turbo'
        }
        
        selected_model_display = st.selectbox(
            'Choose the OpenAI model for extraction:',
            options=list(model_options.keys()),
            index=1,  # Default to GPT-4o Mini
            help="GPT-4o Mini offers the best balance of cost and performance for most document extraction tasks."
        )
        st.session_state.selected_model = model_options[selected_model_display]
        
        # Display model info
        model_info = {
            'gpt-5': "🌟 **Next Generation** - Latest and most advanced model with superior reasoning",
            'gpt-4o-mini': "💡 **Fast & Cost-effective** - Great for most extraction tasks",
            'gpt-4-turbo': "⚡ **Balanced** - Good performance and speed"
        }
        
        if st.session_state.selected_model in model_info:
            st.info(model_info[st.session_state.selected_model])
        
        # File upload - MODIFIED TO ACCEPT MULTIPLE FILES
        st.subheader("📄 Upload PDF Documents")
        uploaded_files = st.file_uploader(
            "Upload one or more PDF documents:", 
            type="pdf", 
            accept_multiple_files=True,  # This enables multiple file upload
            help="You can select and upload multiple PDF files at once"
        )
        
        # Display uploaded files info
        if uploaded_files:
            st.write(f"**📁 {len(uploaded_files)} file(s) uploaded:**")
            for i, file in enumerate(uploaded_files, 1):
                st.write(f"{i}. {file.name}")
        
        # Dynamic fields section
        st.subheader("Fields to Extract")
        st.write("Add the fields you want to extract from the document(s):")
        
        # Text input for adding new fields
        new_field = st.text_input("Enter a field name (e.g., 'Paper Title', 'Authors', 'Methodology'):")
        
        col_add, col_clear = st.columns(2)
        with col_add:
            if st.button("Add Field", use_container_width=True) and new_field.strip():
                if new_field.strip() not in st.session_state.fields_list:
                    st.session_state.fields_list.append(new_field.strip())
                    st.rerun()
        
        with col_clear:
            if st.button("Clear All Fields", use_container_width=True):
                st.session_state.fields_list = []
                st.rerun()
        
        # Display current fields
        if st.session_state.fields_list:
            st.write("**Current fields to extract:**")
            for i, field in enumerate(st.session_state.fields_list):
                col_field, col_remove = st.columns([3, 1])
                with col_field:
                    st.write(f"• {field}")
                with col_remove:
                    if st.button("❌", key=f"remove_{i}"):
                        st.session_state.fields_list.pop(i)
                        st.rerun()
        else:
            st.info("No fields added yet. Add some fields to extract from your document(s).")
        
        # Preset options
        st.subheader("Quick Add Presets")
        
        # Create custom CSS for uniform button height
        st.markdown("""
        <style>
        div[data-testid="stButton"] > button {
            height: 60px;
            white-space: normal;
            word-wrap: break-word;
        }
        </style>
        """, unsafe_allow_html=True)
        
        col_preset1, col_preset2, col_preset3 = st.columns(3)
        
        with col_preset1:
            if st.button("Research Paper", use_container_width=True):
                preset_fields = ["Paper Title", "Authors", "Publication Year", "Abstract", "Methodology", "Results", "Conclusion"]
                for field in preset_fields:
                    if field not in st.session_state.fields_list:
                        st.session_state.fields_list.append(field)
                st.rerun()
        
        with col_preset2:
            if st.button("Business Document", use_container_width=True):
                preset_fields = ["Document Title", "Company Name", "Date", "Executive Summary", "Key Findings", "Recommendations"]
                for field in preset_fields:
                    if field not in st.session_state.fields_list:
                        st.session_state.fields_list.append(field)
                st.rerun()
        
        with col_preset3:
            if st.button("Invoice", use_container_width=True):
                preset_fields = [
                    "Invoice Number",
                    "Invoice Date", 
                    "Due Date",
                    "Billing Information (Customer Name, Address, Contact)",
                    "Seller/Vendor Information (Business Name, Address, Contact, Tax ID)",
                    "Purchase Order (PO) Number",
                    "Description of Goods/Services Provided",
                    "Quantity",
                    "Unit Price",
                    "Line Item Total",
                    "Subtotal",
                    "Taxes (VAT, GST, HST, Sales Tax)",
                    "Discounts",
                    "Shipping/Handling Charges",
                    "Total Amount Due",
                    "Currency",
                    "Payment Terms"
                ]
                for field in preset_fields:
                    if field not in st.session_state.fields_list:
                        st.session_state.fields_list.append(field)
                st.rerun()

    return col1, col2, uploaded_files


# Load the main page
col1, col2, uploaded_files = load_streamlit_page()

# Process the uploaded files
if uploaded_files:
    with col2:
        st.subheader("📑 Document Previews")
        
        # Create document selector instead of tabs for multiple files
        if len(uploaded_files) == 1:
            # Single file - show directly
            st.write(f"**Preview: {uploaded_files[0].name}**")
            display_pdf(uploaded_files[0])
        else:
            # Multiple files - use selectbox instead of tabs
            st.write(f"**Select document to preview ({len(uploaded_files)} files uploaded):**")
            
            # Create options for selectbox
            file_options = [f"📄 {file.name}" for file in uploaded_files]
            selected_file_index = st.selectbox(
                "Choose a document:",
                range(len(uploaded_files)),
                format_func=lambda x: file_options[x],
                label_visibility="collapsed"
            )
            
            # Display the selected PDF
            selected_file = uploaded_files[selected_file_index]
            st.write(f"**Showing: {selected_file.name}**")
            display_pdf(selected_file)
        
    # Process the PDFs and create vector store
    if st.session_state.api_key.strip():
        try:
            # Determine processing method based on OCR configuration
            has_ocr_config = bool(st.session_state.doc_intelligence_endpoint.strip() and st.session_state.doc_intelligence_key.strip())
            
            if has_ocr_config:
                if len(uploaded_files) == 1:
                    processing_message = "🔍 Processing PDF with Azure Document Intelligence OCR..."
                else:
                    processing_message = f"🔍 Processing {len(uploaded_files)} PDFs with Azure Document Intelligence OCR..."
            else:
                if len(uploaded_files) == 1:
                    processing_message = "📄 Processing PDF with traditional parsing..."
                else:
                    processing_message = f"📄 Processing {len(uploaded_files)} PDFs with traditional parsing..."
                    
            with st.spinner(processing_message):
                # Process multiple documents
                all_documents = process_multiple_documents(
                    uploaded_files,
                    doc_intelligence_endpoint=st.session_state.doc_intelligence_endpoint if has_ocr_config else None,
                    doc_intelligence_key=st.session_state.doc_intelligence_key if has_ocr_config else None
                )
                
                # Create collection name based on uploaded files
                if len(uploaded_files) == 1:
                    collection_name = uploaded_files[0].name
                else:
                    collection_name = f"multi_doc_collection_{len(uploaded_files)}_files"
                
                st.session_state.vector_store = create_vectorstore_from_multiple_documents(
                    all_documents, 
                    api_key=st.session_state.api_key,
                    collection_name=collection_name
                )
                
                # Store processed files info
                st.session_state.processed_files = [file.name for file in uploaded_files]
                
            # Success message with processing method used
            if has_ocr_config:
                if len(uploaded_files) == 1:
                    st.success("✅ PDF processed successfully with OCR! Enhanced text extraction completed.")
                else:
                    st.success(f"✅ {len(uploaded_files)} PDFs processed successfully with OCR! Enhanced text extraction completed.")
            else:
                if len(uploaded_files) == 1:
                    st.success("✅ PDF processed successfully with traditional parsing!")
                else:
                    st.success(f"✅ {len(uploaded_files)} PDFs processed successfully with traditional parsing!")
                
        except Exception as e:
            st.error(f"❌ Error processing PDF(s): {str(e)}")
            if "Document Intelligence" in str(e):
                st.info("💡 Try checking your Azure Document Intelligence endpoint and API key, or the service may fall back to traditional PDF parsing.")
    else:
        st.warning("⚠️ Please enter your OpenAI API key to process the PDF(s).")

# Generate extraction results
with col1:
    st.subheader("Extract Information")
    
    # Check if we have everything needed
    can_extract = (
        st.session_state.vector_store is not None and 
        len(st.session_state.fields_list) > 0 and 
        st.session_state.api_key.strip()
    )
    
    if not can_extract:
        missing_items = []
        if not st.session_state.api_key.strip():
            missing_items.append("OpenAI API key")
        if st.session_state.vector_store is None:
            missing_items.append("PDF document(s)")
        if len(st.session_state.fields_list) == 0:
            missing_items.append("extraction fields")
        
        st.warning(f"⚠️ Please provide: {', '.join(missing_items)}")
        
    # Show processing status
    if st.session_state.processed_files:
        st.info(f"📁 **Processed Files ({len(st.session_state.processed_files)}):**")
        for i, filename in enumerate(st.session_state.processed_files, 1):
            st.write(f"{i}. {filename}")
    
    # Show OCR status
    has_ocr_config = bool(st.session_state.doc_intelligence_endpoint.strip() and st.session_state.doc_intelligence_key.strip())
    if uploaded_files and has_ocr_config:
        st.info("🔍 **OCR Enabled**: Using Azure Document Intelligence for enhanced text extraction")
    elif uploaded_files:
        st.info("📄 **Traditional Parsing**: Consider adding Document Intelligence credentials for better text extraction from scanned documents")
    
    # Extract button
    if st.button("Extract Information", disabled=not can_extract):
        if can_extract:
            try:
                with st.spinner("Extracting information from document(s)..."):
                    # Check if we have multiple documents
                    if len(st.session_state.processed_files) > 1:
                        # Use the new per-document extraction function
                        results_per_document = query_document_per_file(
                            vectorstore=st.session_state.vector_store, 
                            fields_list=st.session_state.fields_list,
                            api_key=st.session_state.api_key,
                            model_name=st.session_state.selected_model,
                            document_names=st.session_state.processed_files
                        )
                        
                        st.success("✅ Information extracted successfully!")
                        
                        # Display results for each document separately
                        st.subheader("Extraction Results")
                        
                        # Create a combined CSV for download
                        all_results_for_csv = []
                        
                        for doc_name, result_df in results_per_document.items():
                            st.write(f"### 📄 {doc_name}")
                            
                            # Reorder columns to match the requested format: Document, Field, Answer, Source, Reasoning
                            if not result_df.empty and 'Document' in result_df.columns:
                                ordered_df = result_df[['Document', 'Field', 'Answer', 'Source', 'Reasoning']]
                                st.dataframe(ordered_df, use_container_width=True)
                                
                                # Add to combined results
                                all_results_for_csv.append(ordered_df)
                            else:
                                st.dataframe(result_df, use_container_width=True)
                                all_results_for_csv.append(result_df)
                            
                            st.write("---")  # Separator between documents
                        
                        # Create combined CSV for download
                        if all_results_for_csv:
                            combined_df = pd.concat(all_results_for_csv, ignore_index=True)
                            csv_data = combined_df.to_csv(index=False)
                            
                            download_filename = f"extracted_data_multiple_documents_{len(st.session_state.processed_files)}_files.csv"
                            
                            st.download_button(
                                label="📥 Download All Results as CSV",
                                data=csv_data,
                                file_name=download_filename,
                                mime="text/csv"
                            )
                    
                    else:
                        # Single document - use original function
                        result_df = query_document(
                            vectorstore=st.session_state.vector_store, 
                            fields_list=st.session_state.fields_list,
                            api_key=st.session_state.api_key,
                            model_name=st.session_state.selected_model
                        )
                        
                        st.success("✅ Information extracted successfully!")
                        
                        # Display results
                        st.subheader("Extraction Results")
                        st.dataframe(result_df, use_container_width=True)
                        
                        # Option to download results
                        csv_data = result_df.to_csv()
                        download_filename = f"extracted_data_{st.session_state.processed_files[0].replace('.pdf', '')}.csv"
                        
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv_data,
                            file_name=download_filename,
                            mime="text/csv"
                        )
                
            except Exception as e:
                st.error(f"❌ Error during extraction: {str(e)}")
                st.write("Please check your API key and try again.")

# Add some helpful information in the sidebar
with st.sidebar:
    st.title("📋 How to Use")
    st.markdown("""
    1. **Enter API Keys**: 
       - OpenAI API key (required)
       - Azure Document Intelligence credentials (optional, for OCR)
    2. **Upload PDF(s)**: Select one or multiple PDF documents to analyze
    3. **Add Fields**: Specify what information you want to extract
    4. **Extract**: Click the extract button to get results from all documents
    
    **📄 Multi-Document Features:**
    - Upload multiple PDFs at once
    - Extract information across all documents
    - Results include source document identification
    - Unified search across document collection
    
    **🔍 OCR Benefits:**
    - Better text extraction from scanned documents
    - Improved handling of complex layouts
    - Enhanced accuracy for image-based PDFs
    """)
    
    st.title("🤖 Model Options")
    st.markdown("""
    **GPT-5**: Next generation model with superior reasoning capabilities
    
    **GPT-4o Mini**: Best for most tasks, fast and cost-effective
    
    **GPT-4 Turbo**: Good balance of speed and performance
    """)
    
    st.title("🎯 Tips")
    st.markdown("""
    - Be specific with field names
    - Use natural language descriptions
    - The tool works best with structured documents
    - Results include sources and reasoning
    - For multiple documents, sources will indicate which document contains the information
    - Consider using similar document types for best results
    """)