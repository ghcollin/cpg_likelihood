#include "cpg.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

//#include <iostream>

struct Workspace {
    private:
    CPG::ThreadedWorkspace ws;
    CPG::rIntArray data;
    size_t threads;
    size_t eps_size;

    public:
    Workspace(CPG::constrIntArrayRef data, size_t eps_size, size_t threads) :
        ws(data.maxCoeff(), eps_size, threads),
        data(data),
        threads(threads),
        eps_size(eps_size)
    {
    }

    std::tuple<CPG::rIntArray, size_t, size_t> getstate() const {
        return std::tuple<CPG::rIntArray, size_t, size_t>(this->data, this->eps_size, this->threads);
    }

    std::tuple<double, bool> llh(const std::vector<CPG::Model>& models, CPG::constrArrayRef lambda_array) {
        return CPG::llh_threaded(this->ws, this->data, models, lambda_array);
    }
};

namespace py = pybind11;

PYBIND11_MODULE(llh, m) {
    py::class_<Workspace>(m, "Workspace")
        .def(
            py::init<CPG::constrIntArrayRef, size_t, size_t>(), 
            py::arg("data").noconvert(), 
            py::arg("eps_size").noconvert(),
            py::arg("threads") = 1)
        .def(
            "eval", &Workspace::llh,
            py::arg("models"),
            py::arg("lambda").noconvert())
        .def("__getstate__", &Workspace::getstate)
        .def("__setstate__",
            [](Workspace& ws, const std::tuple<CPG::rIntArray, size_t, size_t>& t) { // __setstate__
                new (&ws) Workspace(std::get<0>(t), std::get<1>(t), std::get<2>(t));
            }
        )
        ;
    py::class_<CPG::Model>(m, "Model")
        .def(
            py::init<double, CPG::constrArrayRef, CPG::constrArrayRef, CPG::constrArrayRef, CPG::constMatrixRef>(),
            py::arg("A").noconvert(),
            py::arg("Fbs").noconvert(),
            py::arg("ns").noconvert(),
            py::arg("eps").noconvert(),
            py::arg("d_mu_eps").noconvert(),
            py::keep_alive<1, 3>(),
            py::keep_alive<1, 4>(),
            py::keep_alive<1, 5>(),
            py::keep_alive<1, 6>()
        )
        ;
}