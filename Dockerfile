FROM python:3.10-slim as build
WORKDIR /src
ENV PYTHONBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1
EXPOSE 8501
COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get -y install curl
RUN apt-get -y install libgomp1 build-essential libssl-dev libffi-dev libjpeg-dev zlib1g-dev libgfortran5 tcl tcl-dev
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY . .
COPY --chmod=0755 ./bin/opensees.debian /usr/local/bin/opensees
ENTRYPOINT ["streamlit", "run", "lab.py", "--server.port=8501", "--server.address=0.0.0.0"]

