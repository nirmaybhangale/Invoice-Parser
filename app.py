import streamlit as st
from Summarizer import extract_text, generate_invoice_summary

st.set_page_config(page_title="Hybrid Invoice AI", page_icon="📑")
st.title("📑 Universal Invoice Parser")
st.write("Upload a PDF, JPG, or PNG invoice for AI analysis.")

uploaded_file = st.file_uploader("Choose a file", type=["pdf", "jpg", "jpeg", "png"])

if uploaded_file is not None:
    if "image" in uploaded_file.type:
        st.image(uploaded_file, caption="Uploaded Image", width=300)

    if st.button("Analyze Document"):
        with st.spinner("Processing (this may take a moment for OCR)..."):
            try:
                #Extract
                raw_text = extract_text(uploaded_file)
                
                #Summarize
                result = generate_invoice_summary(raw_text)
                
                #Display
                st.subheader("Extracted Details")
                st.success("Done!")
                st.info(result)
                
                with st.expander("Show Raw Extracted Text"):
                    st.write(raw_text)
                    
            except Exception as e:
                st.error("Something went wrong!")
                st.exception(e)