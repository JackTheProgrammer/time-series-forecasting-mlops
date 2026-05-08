# 1. Base Image
FROM python:3.11-slim-bookworm

# 2. Set Working Directory
WORKDIR /project

# 3. Copy ONLY requirements first to leverage Docker cache
COPY requirements.txt .

# 4. Optimization: Install CPU-only Torch to reduce size (from 3GB to ~700MB)
# If you need GPU, keep your current requirements but expect the long wait.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.0 \
    torchvision==0.18.0

# 5. Install the rest of the requirements
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the project
COPY . .

# 7. Create a startup script (The "Process Manager")
# This runs the API in the background (&) and Streamlit in the foreground
RUN echo "#!/bin/sh\n\
python scripts/api/main.py &\n\
streamlit run scripts/app/home/home.py --server.port=8501 --server.address=0.0.0.0" > /project/start.sh

# 8. Give execution permission
RUN chmod +x /project/start.sh

# 9. Expose both ports (API usually 5050, Streamlit 8501)
EXPOSE 5050
EXPOSE 8501

# 10. Launch the script
CMD ["/project/start.sh"]