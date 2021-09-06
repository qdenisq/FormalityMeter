# Using lightweight image
FROM python:3.9-slim

# Installing packages
# RUN apt-get update
# Defining working directory and adding source code
WORKDIR /usr/src/app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY bootstrap.sh ./
COPY src ./src

# Start app
EXPOSE 5050
ENTRYPOINT ["/usr/src/app/bootstrap.sh"]