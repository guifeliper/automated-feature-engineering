FROM smizy/scikit-learn:latest

# EXPOSE 5000
# EXPOSE 8888

WORKDIR /app

COPY requirements.txt /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN apk add --update --no-cache graphviz

# RUN conda install scikit-learn

# COPY . /app
