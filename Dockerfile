#FROM debian:latest
FROM ubuntu:18.04
#  $ docker build . -t continuumio/miniconda3:latest -t continuumio/miniconda3:4.5.11
#  $ docker run --rm -it continuumio/miniconda3:latest /bin/bash
#  $ docker push continuumio/miniconda3:latest
#  $ docker push continuumio/miniconda3:4.5.11


ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH


RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git gunicorn libgl1-mesa-glx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

ADD . /myapp

WORKDIR /myapp
ADD requirements.txt /myapp/requirements.txt
ADD ubuntu-requirements.txt /myapp/ubuntu-requirements.txt
#ADD conda-requirements_broken.txt /myapp/conda-requirements_broken.txt

RUN conda install --file /myapp/ubuntu-requirements.txt --yes --channel conda-forge
#RUN conda install --file /myapp/conda-requirements_broken.txt --yes --channel conda-forge/label/broken opencv
RUN pip install -r /myapp/requirements.txt

#RUN apt-get install -y libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
#RUN apt install -y build-essential cmake git libgtk2.0-dev


# ENV TINI_VERSION v0.16.1
# ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
# RUN chmod +x /usr/bin/tini

#ENTRYPOINT [ "/usr/bin/tini", "--" ]
#EXPOSE 8080
#CMD [ "/bin/bash" ]
#CMD ["gunicorn", "-b","0.0.0.0:8080", "main:server", "-t", "3600"]
CMD exec gunicorn -b :$PORT main:server --timeout 1800
