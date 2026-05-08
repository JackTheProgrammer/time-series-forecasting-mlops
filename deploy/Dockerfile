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

# 6. Finally, copy the rest of the project
COPY . .

# 7. Running the crafted RESTful API
RUN python scripts/api/main.py

# 8. Expose Streamlit port and the running the streamlit app via CMD
EXPOSE 8501

CMD ["streamlit", "run", "scripts/app/home.py"]