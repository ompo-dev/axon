# Corpo do Toshi em container. Build a partir da RAIZ axon/ (precisa do núcleo):
#   docker build -f plataforma/Dockerfile -t toshi .
#   docker run -p 8770:8770 -v "%cd%/core_v2_prototype/dados:/axon/core_v2_prototype/dados" toshi
# (monte 'dados' pra levar a memória salva / os livros; câmera/mic ficam NO NAVEGADOR, não no container.)
FROM python:3.13-slim
WORKDIR /axon
RUN pip install --no-cache-dir numpy
COPY core_v2_prototype ./core_v2_prototype
COPY plataforma ./plataforma
EXPOSE 8770
CMD ["python", "plataforma/servidor.py"]
