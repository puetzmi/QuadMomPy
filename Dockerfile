FROM python:3.10.6-slim
#RUN apk add --update make cmake gcc g++ gfortran
#RUN apk add --update python3-dev
#RUN pip3 install cython
#RUN pip3 install numpy

COPY . /quadmompy
WORKDIR /quadmompy
RUN pip3 install -r requirements.txt

# For some reason installation of SciPy fails when installing 
# from requirements file directly 
#RUN for pkg in $(sed -e '/^#/d' requirements.txt | grep -v scipy); do pip3 install $pkg; done
#RUN for pkg in $(sed -e '/^#/d' requirements.txt | grep scipy); do pip3 install $pkg; done

ENTRYPOINT [ "bash" ]