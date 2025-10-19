FROM python:3.9-slim

WORKDIR /app


COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    libpq-dev \
    libpq5 \
    gcc \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip 

RUN pip install --no-cache-dir -r requirements.txt

# Download required NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
