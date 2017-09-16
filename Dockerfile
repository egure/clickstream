FROM alpine:3.5

# Install python and pip
RUN apk add --update py2-pip

# upgrade pip
RUN pip install --upgrade pip

# install Python modules needed by the Python app

RUN pip install --upgrade -r requirements.txt

# tell the port number the container should expose
EXPOSE 5000

# run the application
CMD ["python", "/home/gr/Escritorio/clickstream/app/app.py"]
