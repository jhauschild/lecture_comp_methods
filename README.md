# Compuational Methods in Many-Body Physics

Authors: Prof. Frank Pollmann, Prof. Michael Knap, Johannes Hauschild

This is a set of tutorial codes and exercises for the lecture *Compuational Methods in Many-Body Physics* given at TU Munich in the summer term 2018.
Lecture notes, exercise sheets and further references are available at the course web page: https://www.cmt.ph.tum.de/index.php?id=65
The codes are organized in folders by topics, since some of the exercises build on previous codes.

For some exercise, we provide some codes implementing various algorithms, as pure python 3 files with `*.py` filenames,
and some data files, included in the folder `exercises`.
The solutions to the exercise are provided as jupyter notebooks (formerly known as ipython notebooks), with `sol*.ipynb` filenames.
If the exercise asked to modify a function, it is copied into the notebook, leaving the original templates `*.py` untouched as provided on the course webpage.

## Setup
If you are completely new to python, a good introduction for our course are the scipy lectures

If you have a laptop, a very good python distribution is provided by Anaconda, available at https://www.anaconda.com/download. It ships with jupyter, uses Intel MKL and comes with the numba package. 

The [numba package](http://numba.pydata.org/) is used in some of the codes for optimization.
This brings in some cases a speed up of up to 100, installing numba is therefore highly recommended. 
If you really have big trouble installing numba, or find the error messages produced by it too confusing, 
you can copy the file `numba.py` providing a dummy `@jit` decorator into the other folders. 
The price is that you loose the speed-up...

On the work stations in the CIP pool at TUM, the intelpython distribution is installed.
To start a jupyter notebook, follow these steps:
1. Open a terminal
2. Go to the directory where you keep your scripts/notebooks using `cd some/directory`
3. On your laptop, enter `jupyter notebook`, or at the TUM workstations enter `/mount/packs/intelpython35/bin/jupyter notebook`
   This should start a local server opening a webpage in the browser (e.g. firefox), where you can create python notbooks etc.
   If you close the web page by accident, open the page http://localhost:8888/ 
4. Do your calculations.
5. Once you're done, stop the server in the terminal by pressing Ctrl-C and confirm with 'y'

We use the ipython magic `%matplotlib inline` to include plots showing the results directly in the notebooks.

# Further references

- Introduction to Python, https://www.scipy-lectures.org/
- Lecture Notes by Anders W. Sandvik, http://arxiv.org/abs/1101.3281v1
- Lecture Notes by Johannes Hauschild, Frank Pollmann, https://arxiv.org/abs/1805.00055
- Review on DMRG by Ulrich Schollwoeck, http://arxiv.org/abs/1008.3477
- Online book by Michael Nielsen, http://neuralnetworksanddeeplearning.com
