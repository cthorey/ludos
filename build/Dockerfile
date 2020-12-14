ARG CUDAGL_TAG
FROM nvidia/cudagl:$CUDAGL_TAG

# change shell
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# apt install
COPY build/apt /build/apt
RUN apt update && \
  cat /build/apt | xargs apt install --allow-unauthenticated --yes && \
  apt clean

# install our library
COPY . /workdir
RUN pip3 install -r /workdir/build/requirements.txt
RUN pip3 install -e /workdir/

# set up ipython
COPY build/ipython_config.py /root/.ipython/profile_default/ipython_config.py
COPY build/jupyter_notebook_config.py /root/.jupyter/

# set up env
ENV ROOT_DIR=/workdir
ENV PASSWORD=tf
ENV PATH=$PATH:/workdir/scripts

WORKDIR /workdir

CMD ["bash"]
