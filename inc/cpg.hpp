#ifndef CPG_HPP
#define CPG_HPP

#include <vector>
#include <exception>
#include "Eigen/Dense"

namespace CPG {

using Vector = Eigen::VectorXd;
using rVector = Eigen::RowVectorXd;
using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using cArray = Eigen::Array<double, Eigen::Dynamic, 1>;
using rArray = Eigen::Array<double, 1, Eigen::Dynamic>;
using rIntArray = Eigen::Array<int64_t, 1, Eigen::Dynamic>;

using VectorRef = Eigen::Ref<Vector>;
using constVectorRef = Eigen::Ref<const Vector>;
using rVectorRef = Eigen::Ref<rVector>;
using constrVectorRef = Eigen::Ref<const rVector>;
using MatrixRef = Eigen::Ref<Matrix>;
using constMatrixRef = Eigen::Ref<const Matrix>;
using cArrayRef = Eigen::Ref<cArray>;
using constcArrayRef = Eigen::Ref<const cArray>;
using rArrayRef = Eigen::Ref<rArray>;
using constrArrayRef = Eigen::Ref<const rArray>;
using rIntArrayRef = Eigen::Ref<rIntArray>;
using constrIntArrayRef = Eigen::Ref<const rIntArray>;

struct SeriesWorkspace {
    public:
    SeriesWorkspace(int64_t max_power, size_t eps_start, size_t eps_size) :
        max_power(max_power),
        eps_start(eps_start),
        eps_size(eps_size),
        scratch(eps_size),
        models_power_series()
    {
    }

    struct Scratch {
        public:
        Scratch(size_t eps_size) :
            J0(eps_size),
            ln_J0(eps_size),
            scratch1(eps_size),
            scratch2(eps_size),
            scratch3(eps_size)
        {

        }

        rArray J0, ln_J0, scratch1, scratch2, scratch3;
    };

    int64_t max_power;
    size_t eps_start;
    size_t eps_size;
    Scratch scratch;
    std::vector<Matrix> models_power_series;

    void add_power_series() {
        this->models_power_series.emplace_back(this->max_power+1, this->eps_size);
    }
};

struct LLHWorkspace {
    public:
    LLHWorkspace(int64_t max_power) :
        max_power(max_power),
        z_series(max_power+1),
        ln_z_series(max_power+1),
        ln_ps(max_power+1)
    {
    }

    int64_t max_power;
    rArray z_series, ln_z_series, ln_ps;
};

struct Model {
    public:
    Model(double A, constrArrayRef Fbs, constrArrayRef ns, constrArrayRef eps, constMatrixRef d_mu_eps) :
        A(A),
        Fbs(Fbs),
        ns(ns),
        eps(eps),
        d_mu_eps(d_mu_eps)
    {
        if (Fbs.cols() < 1) {
            throw std::runtime_error("During construction of model: number of flux breaks less than one, must be one or more.");
        }
        if (ns.cols() != Fbs.cols()+1) {
            throw std::runtime_error("During construction of model: number of indicies != number of flux breaks + 1. len(ns) = " + std::to_string((int)ns.cols()) + ", len(Fbs) = " + std::to_string((int)Fbs.cols()));
        }
    }

    double A;
    constrArrayRef Fbs;
    constrArrayRef ns;

    constrArrayRef eps;
    constMatrixRef d_mu_eps;
};

struct Workspace {
    public:
    Workspace(int64_t max_power, size_t eps_size) :
        llh(max_power),
        series(max_power, 0, eps_size) {

    }
    
    LLHWorkspace llh;
    SeriesWorkspace series;
};

struct ThreadedWorkspace {
    public:
    ThreadedWorkspace(int64_t max_power, size_t eps_size, size_t threads) :
        llh_spaces(),
        series_spaces() {
        
        size_t eps_chunk = std::ceil(((double)eps_size)/((double)threads));
        for (size_t i = 0; i < threads; i++) {
            llh_spaces.emplace_back(max_power);
            auto eps_start = i*eps_chunk;
            auto eps_end = std::min(eps_start + eps_chunk, eps_size);
            auto ws_size = eps_end - eps_start;
            series_spaces.emplace_back(max_power, eps_start, ws_size);
        }
    }

    std::vector<LLHWorkspace> llh_spaces;
    std::vector<SeriesWorkspace> series_spaces;
};

std::tuple<double, bool> llh(Workspace ws, constrIntArrayRef data, const std::vector<Model>& models, constrArrayRef lambda_array);

std::tuple<double, bool> llh_threaded(ThreadedWorkspace& threads_ws, constrIntArrayRef data, const std::vector<Model>& models, constrArrayRef lambda_array);

}

#endif