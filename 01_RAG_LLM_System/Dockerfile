FROM python:3.11.7-slim-bookworm

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# # Install necessary dependencies. Post Git additional packages are for OpenCV
# RUN apt-get update && apt-get install -y curl gnupg git 

# # Open CV issues
# RUN apt-get install ffmpeg libsm6 libxext6  -y

# # Update the package lists
# RUN apt-get update && \
#     apt-get install -y wget gnupg

# # Install Chromium
# RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - && \
#     echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list && \
#     apt-get update && \
#     apt-get install -y google-chrome-stable

WORKDIR /code    

COPY . .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn" ,"app:app", "--host=0.0.0.0","--port=8000",  "--workers=4"]


