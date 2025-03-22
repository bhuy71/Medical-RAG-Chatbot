#!/bin/bash

# Check and create necessary directories
mkdir -p data model vectorstore/medical_faiss static templates

# Copy style.css to static directory if it doesn't exist
if [ ! -f "static/style.css" ]; then
    echo "Copying style.css to static directory..."
    cp style.css static/
fi

# Copy chat.html to templates directory if it doesn't exist
if [ ! -f "templates/chat.html" ]; then
    echo "Copying chat.html to templates directory..."
    cp chat.html templates/
fi

# Check if there are PDFs in the data directory
pdf_count=$(ls -1 data/*.pdf 2>/dev/null | wc -l)
if [ $pdf_count -eq 0 ]; then
    echo "WARNING: No PDF files found in 'data/' directory. Please add PDF documents before continuing."
    exit 1
fi

# Check Llama2 model
if [ ! -f "model/llama-2-7b-chat.Q4_0.gguf" ]; then
    echo "WARNING: Llama2 model not found. Please download llama-2-7b-chat.Q4_0.gguf and place it in the 'model/' directory."
    echo "You can download the model from: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/tree/main"
    exit 1
fi

# Install required libraries
echo "Installing required libraries..."
pip install -r requirements.txt

# Create vector database
echo "Creating vector database..."
python store_index.py

echo "Setup complete. You can run the application with: python app.py"