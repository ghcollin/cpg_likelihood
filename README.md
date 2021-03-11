# Downloading

To download, use the following command to clone the repository and all submodules:

    git clone --recurse-submodules https://github.com/ghcollin/cpg_likelihood.git

# Installation

You will need to install the GNU Scientific Library (GSL).

# Dependencies

In addition to GSL, this library makes use of the Eigen C++ linear algebra library.
A recent development version of Eigen is linked into this repository though a git submodule, as the latest stable version does not include features required by this library.
A modified version of the expoential integral algorithm written by [Guillermo Navas-Palencia](https://gnpalencia.org) and described in [this paper](https://gnpalencia.org/research/GNP_Expint2017.pdf) is included in this repository. 
The algorithm has been modified to make it more numerically stable for this specific application, where catastrophic cancellation can occur when the source flux is small.

# References

If you use this library, please cite

    @article{navas-palenciaFastAccurateAlgorithm2018,
        title = {Fast and Accurate Algorithm for the Generalized Exponential Integral {{E}} ν (x) for Positive Real Order},
        author = {Navas-Palencia, Guillermo},
        date = {2018-02},
        journaltitle = {Numerical Algorithms},
        shortjournal = {Numer Algor},
        volume = {77},
        pages = {603--630},
        issn = {1017-1398, 1572-9265},
        doi = {10.1007/s11075-017-0331-z},
        url = {http://link.springer.com/10.1007/s11075-017-0331-z},
        urldate = {2020-10-27},
        langid = {english},
        number = {2}
    }

    [ todo: insert our paper ]