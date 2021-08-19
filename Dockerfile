FROM python:3.6-slim

# RUN pip install --no-cache-dir pipenv
WORKDIR /app
# this is where we put all the prediction results
VOLUME [ "/log" ]

# pip way
COPY ./ /app
RUN pip install .

# use entrypoint so we could pass arguments to the container
ENTRYPOINT [ "trident" ]

# see Usage section in README.md file
