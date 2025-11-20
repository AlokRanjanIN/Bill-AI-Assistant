import streamlit as st
import pandas as pd
import json
import os
import tempfile
from io import BytesIO
from PIL import Image
from assistant import BillAssistant  # assume this is the provided class
import hashlib

# Cache the assistant so models load only once
@st.cache_resource
def load_assistant():
    return BillAssistant(use_colab_secrets=True)

assistant = load_assistant()

st.set_page_config(page_title="Bill Assistant", layout="centered")
st.title("Bill Assistant")
st.write("Upload a bill image (PNG/JPEG) or PDF below and then click **Process bill**. Processing will run once per uploaded file unless you clear it.")

# Initialize session state keys we use
if "uploaded_file_id" not in st.session_state:
    st.session_state.uploaded_file_id = None
if "temp_path" not in st.session_state:
    st.session_state.temp_path = None
if "process_result" not in st.session_state:
    st.session_state.process_result = None
if "processed_ok" not in st.session_state:
    st.session_state.processed_ok = False
if "messages" not in st.session_state:
    st.session_state.messages = []

def make_file_id(uploaded_file) -> str:
    """Create a simple id for uploaded file to avoid reprocessing same content."""
    # Use filename + size + sha1 of bytes for more certainty
    try:
        content = uploaded_file.getbuffer()
        h = hashlib.sha1(content).hexdigest()
        return f"{uploaded_file.name}-{uploaded_file.size}-{h}"
    except Exception:
        return f"{uploaded_file.name}-{uploaded_file.size}"

def save_uploaded_to_temp(uploaded_file) -> str:
    """Save Streamlit uploaded file to a unique temp path and return the path."""
    suffix = ""
    if uploaded_file.type == "application/pdf" or uploaded_file.name.lower().endswith(".pdf"):
        suffix = ".pdf"
    else:
        # try to preserve extension
        ext = os.path.splitext(uploaded_file.name)[1] or ".png"
        suffix = ext if ext.startswith(".") else f".{ext}"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="bill_")
    with open(tmp.name, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return tmp.name

# File uploader for image or PDF bills
uploaded_file = st.file_uploader("Choose an image (PNG/JPEG) or PDF", type=["png", "jpg", "jpeg", "pdf"])

if uploaded_file is not None:
    file_id = make_file_id(uploaded_file)
    if uploaded_file.type != "application/pdf":
        try:
            preview_image = Image.open(uploaded_file).convert("RGB")
            print(f"Previewing image...")
            st.image(preview_image, caption="Preview", width="content")
        except Exception:
            st.write("Preview not available for this image.")
    else:
        st.write("PDF uploaded. Preview not shown.")

    # If it's the same file already processed, show status and results without reprocessing
    if st.session_state.uploaded_file_id == file_id and st.session_state.processed_ok:
        st.success("This file was already processed in this session.")
    else:
        # Show action buttons
        cols = st.columns([1, 1, 1])
        with cols[0]:
            if st.button("Process bill"):
                # save to temp and process
                tmp_path = save_uploaded_to_temp(uploaded_file)
                st.session_state.temp_path = tmp_path
                st.session_state.uploaded_file_id = file_id

                # run processing with a spinner
                with st.spinner("Processing bill (OCR + parsing)... this may take a while for large PDFs"):
                    # option: free GPU memory (harmless if not available)
                    try:
                        import torch, gc
                        torch.cuda.empty_cache()
                        gc.collect()
                    except Exception:
                        pass

                    result = assistant.process_bill(tmp_path)
                    st.session_state.process_result = result
                    st.session_state.processed_ok = result.startswith("✅")

                if st.session_state.processed_ok:
                    st.success("Bill processed successfully!")
                else:
                    st.error("Processing failed: " + (st.session_state.process_result or "Unknown error"))

        with cols[1]:
            if st.button("Clear / Reset"):
                # remove temp file if any
                try:
                    if st.session_state.temp_path and os.path.exists(st.session_state.temp_path):
                        os.remove(st.session_state.temp_path)
                except Exception:
                    pass
                # reset session-state fields
                st.session_state.uploaded_file_id = None
                st.session_state.temp_path = None
                st.session_state.process_result = None
                st.session_state.processed_ok = False
                st.session_state.messages = []
                st.experimental_rerun()

        with cols[2]:
            if st.button("Show raw extracted text (if processed)"):
                if st.session_state.processed_ok and assistant.bill_text:
                    st.text_area("Extracted text", assistant.bill_text, height=300)
                else:
                    st.info("No extracted text available. Process the file first.")

# If processing was successful, show summary & download
if st.session_state.processed_ok and assistant.current_bill:
    bill = assistant.current_bill
    st.write("---")
    st.header("Extracted bill summary")
    st.markdown(f"**Invoice:** {bill.get('invoice_number', 'N/A')}")
    st.markdown(f"**Date:** {bill.get('invoice_date', 'N/A')}")
    st.markdown(f"**Total:** ${bill.get('total', 'N/A')}")

    items = bill.get("items", [])
    if items:
        df_items = pd.DataFrame(items)
        st.write("**Line Items:**")
        st.table(df_items)

    bill_json = json.dumps(bill, indent=2)
    st.download_button(
        "Download Bill as JSON",
        data=bill_json,
        file_name="extracted_bill.json",
        mime="application/json"
    )

# Chat interface for queries about the bill
st.write("---")
st.write("### Ask questions about the bill or chit-chat")

# Display previous messages
for msg in st.session_state.messages:
    role = msg.get("role", "user")
    with st.chat_message(role):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Your message...")
if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    # Get assistant response
    response = assistant.chat(user_input)
    st.session_state.messages.append({"role": "assistant", "content": response})
    # Display assistant's new message
    with st.chat_message("assistant"):
        st.markdown(response)
