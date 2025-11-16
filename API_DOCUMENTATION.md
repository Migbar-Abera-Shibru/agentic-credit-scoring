# Agentic Credit Scoring API Documentation

## Base URL
`http://localhost:8000`

## Endpoints

### 1. Health Check
**GET** `/health`
- Check API status and model loading

### 2. Basic Prediction  
**POST** `/predict`
- Input: Borrower features
- Output: Probability (0-1)

### 3. Agentic Analysis
**POST** `/agent` 
- Input: Borrower features + natural language query
- Output: Probability, reasoning, risk factors, recommendations

## Example Usage

### Python
```python
import requests

response = requests.post("http://localhost:8000/agent", json={
    "features": {
        "RevolvingUtilizationOfUnsecuredLines": 0.5,
        "age": 35,
        # ... other features
    },
    "query": "Explain my risk and suggest improvements"
})

curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": {...}}'

### **4. 🐳 Create Docker Configuration** (Optional)

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]