FROM tensorflow/tensorflow:2.8.0
RUN pip3 install graphviz sklearn networkx matplotlib ipython
COPY . /project2/
WORKDIR /project2
RUN mkdir -p data
ENTRYPOINT [ "python3", "training.py"]