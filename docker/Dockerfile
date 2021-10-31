FROM julia

ARG USER_ID
ARG GROUP_ID
RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
# Root password
RUN echo 'root:root' | chpasswd
RUN apt-get update
RUN apt-get install -y unzip bzip2

# RUN chown -R user:user /home/user/project/env
USER user
WORKDIR /home/user
RUN mkdir  env
COPY --chown=user:user Project.toml env/Project.toml
# COPY --chown=user:user Manifest.toml env/Manifest.toml
RUN julia  -e 'using Pkg; Pkg.activate("env"); Pkg.instantiate();'
RUN julia  -e 'using Pkg; Pkg.activate("env"); using Conda; Conda.add("jupyter"); Conda.add("matplotlib"); Pkg.precompile();'
RUN mkdir /home/user/.jupyter/
RUN echo "c.NotebookApp.ip = '0.0.0.0'\nc.NotebookApp.token = ''" >> /home/user/.jupyter/jupyter_notebook_config.py

# mosek licence
RUN mkdir /home/user/mosek
COPY --chown=user:user mosek.lic /home/user/mosek/
