# Use an official Python runtime as a parent image
FROM continuumio/miniconda3

RUN apt-get --allow-releaseinfo-change update
RUN apt-get install -y build-essential 

RUN mkdir /tmp/cansrmapp
COPY ./ /tmp/cansrmapp/
RUN pip install /tmp/cansrmapp

RUN rm -rf /tmp/cansrmapp

ENTRYPOINT ["/opt/conda/bin/cansrmappcmd.py"]

CMD ["--help"]
