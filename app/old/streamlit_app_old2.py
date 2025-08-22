import streamlit as st  
from functions_old2 import *
import base64

# Initialize session state variables
if 'api_key' not in st.session_state:
    st.session_state.api_key = ''

if 'fields_list' not in st.session_state:
    st.session_state.fields_list = []

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

def display_pdf(uploaded_file):
    """
    Display a PDF file that has been uploaded to Streamlit.

    The PDF will be displayed in an iframe, with the width and height set to 700x1000 pixels.

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
    
    # Embed PDF in HTML
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    
    # Display file
    st.markdown(pdf_display, unsafe_allow_html=True)


def load_streamlit_page():
    """
    Load the streamlit page with two columns. The left column contains a text input box for the user to input their OpenAI API key, and a file uploader for the user to upload a PDF document. The right column contains a header and text that greet the user and explain the purpose of the tool.

    Returns:
        col1: The left column Streamlit object.
        col2: The right column Streamlit object.
        uploaded_file: The uploaded PDF file.
    """
    st.set_page_config(layout="wide", page_title="Dynamic PDF Extraction Tool")

    # Design page layout with 2 columns: File uploader on the left, and other interactions on the right.
    col1, col2 = st.columns([0.5, 0.5], gap="large")

    with col1:
        st.header("Configuration")
        
        # API Key input
        st.subheader("OpenAI API Key")
        st.text_input('Enter your OpenAI API key:', type='password', key='api_key',
                    label_visibility="collapsed", disabled=False)
        
        # File upload
        st.subheader("Upload PDF Document")
        uploaded_file = st.file_uploader("Please upload your PDF document:", type="pdf")
        
        # Dynamic fields section
        st.subheader("Fields to Extract")
        st.write("Add the fields you want to extract from the document:")
        
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
            st.info("No fields added yet. Add some fields to extract from your document.")
        
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

    return col1, col2, uploaded_file


# Load the main page
col1, col2, uploaded_file = load_streamlit_page()

# Process the uploaded file
if uploaded_file is not None:
    with col2:
        st.subheader("PDF Preview")
        display_pdf(uploaded_file)
        
    # Process the PDF and create vector store
    if st.session_state.api_key.strip():
        try:
            with st.spinner("Processing PDF..."):
                documents = get_pdf_text(uploaded_file)
                st.session_state.vector_store = create_vectorstore_from_texts(
                    documents, 
                    api_key=st.session_state.api_key,
                    file_name=uploaded_file.name
                )
            st.success("✅ PDF processed successfully!")
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
    else:
        st.warning("⚠️ Please enter your OpenAI API key to process the PDF.")

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
            missing_items.append("PDF document")
        if len(st.session_state.fields_list) == 0:
            missing_items.append("extraction fields")
        
        st.warning(f"⚠️ Please provide: {', '.join(missing_items)}")
    
    # Extract button
    if st.button("Extract Information", disabled=not can_extract):
        if can_extract:
            try:
                with st.spinner("Extracting information..."):
                    result_df = query_document(
                        vectorstore=st.session_state.vector_store, 
                        fields_list=st.session_state.fields_list,
                        api_key=st.session_state.api_key
                    )
                
                st.success("✅ Information extracted successfully!")
                
                # Display results
                st.subheader("Extraction Results")
                st.dataframe(result_df, use_container_width=True)
                
                # Option to download results
                csv_data = result_df.to_csv()
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv_data,
                    file_name=f"extracted_data_{uploaded_file.name.replace('.pdf', '')}.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"❌ Error during extraction: {str(e)}")
                st.write("Please check your API key and try again.")

# Add some helpful information in the sidebar
with st.sidebar:
    st.title("📋 How to Use")
    st.markdown("""
    1. **Enter API Key**: Add your OpenAI API key
    2. **Upload PDF**: Select a PDF document to analyze
    3. **Add Fields**: Specify what information you want to extract
    4. **Extract**: Click the extract button to get results
    
    **Field Examples:**
    
    📄 **Research Papers:**
    - Paper Title, Authors, Publication Date
    - Abstract, Methodology, Results
    
    💼 **Business Documents:**
    - Company Name, Executive Summary
    - Key Findings, Recommendations
    
    🧾 **Invoices:**
    - Invoice Number, Invoice Date, Due Date
    - Billing Information, Vendor Information
    - Total Amount Due, Payment Terms
    """)
    
    st.title("🎯 Tips")
    st.markdown("""
    - Be specific with field names
    - Use natural language descriptions
    - The tool works best with structured documents
    - Results include sources and reasoning
    """)