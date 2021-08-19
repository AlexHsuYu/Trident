FROM python:3.6-stretch

WORKDIR /app

RUN pip install numpy==1.14.5 \
                pandas \
                scipy \
                scikit-learn \
                click

# mount your app into this make-shift container

CMD [ "bash" ]

# docker build -t trident-dev .
# docker run -it --rm -v $PWD:/app trident-dev
# docker exec -it trident-dev bash
