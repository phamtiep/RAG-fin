# Use official Python image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements file first for better caching
COPY app/requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy application code
COPY app/ .



# Expose port for the application
EXPOSE 8501

# Command to run the application

#CMD ["bash", "-c", "python scrapper/main.py && streamlit run streamlit.py --server.port=8501"]
 CMD ["bash", "-c", " streamlit run streamlit.py --server.port=8501"]