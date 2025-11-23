@REM docker build -t my-python .
@REM docker run --rm -v "%cd%":/workspace my-python python --version*


docker run -it -v "%cd%":/workspace -w /workspace my-python ls
docker run -it -v "%cd%":/workspace -w /workspace my-python python one_neuron_simulation.py