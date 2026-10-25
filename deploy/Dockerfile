# Runs anywhere Docker does (deployed on Azure Container Apps; also works on Hugging Face Spaces). The app listens on
# $PORT if the host sets it, otherwise 7860.
FROM python:3.11-slim
LABEL org.opencontainers.image.source="https://github.com/Janacreates/skin-lesion-shift-peft"

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
