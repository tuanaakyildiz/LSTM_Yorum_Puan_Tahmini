# 1. Start with a lightweight Python image
FROM python:3.12-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file and install dependencies
# (Make sure gunicorn is added to your requirements.txt!)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of your Python files (model, app.py, tokenizer)
# (Thanks to .dockerignore, this will NOT copy the React folder)
COPY . .

# 5. Expose the port Flask uses
EXPOSE 7860

# 6. Start the Flask server using Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]