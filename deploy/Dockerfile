# Runs anywhere Docker does (Azure Container Apps, Hugging Face Spaces, Render). The app listens on
# $PORT if the host sets it, otherwise 7860.
FROM python:3.11-slim

# A non-root user (Hugging Face Spaces require uid 1000; harmless elsewhere).
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app
ENV PORT=7860

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

EXPOSE 7860
CMD ["python", "app.py"]
