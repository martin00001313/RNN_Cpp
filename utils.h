#include <cmath>
#include <array>
#include <random>

namespace NN
{

using f_type = double;

template<size_t N>
using sample_type = std::pair<std::array<f_type, N>, f_type>;

template<size_t N, size_t M>
using array2D = std::array<std::array<f_type, M>, N>;

template <size_t N>
constexpr std::array<f_type, N> get_waves(size_t start_point) noexcept
{
    std::array<f_type, N> data;

    for (size_t i = start_point; i < N + start_point; ++i) {
        data[i] = std::sin(i);
    }

    return data;
}

template<size_t N, size_t seq_length>
std::array<sample_type<seq_length>, N> get_data(size_t start_point) noexcept
{
    std::array<sample_type<seq_length>, N> data;

    for (size_t i = 0; i < N; ++i) {
        data[i].first = get_waves<seq_length>(i);
        data[i].second = get_waves<1>(i + seq_length)[0];
    }

    return data;
}

template<size_t N, size_t M>
inline array2D<N, M> get_weights() noexcept
{
    array2D<N, M> data;
    std::random_device rd;
    std::mt19937_64 mt_gen(rd());
    std::uniform_real_distribution<f_type> dist;

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < M; ++j) {
            data[i][j] = dist(mt_gen);
        }
    }
    return data;
}

constexpr f_type sigmoid(f_type val) noexcept
{
    constexpr f_type one = 1.;
    return one/(one + (f_type)std::exp(-val));
}

template<size_t N>
constexpr std::array<f_type, N> sigmoid(const std::array<f_type, N>& vec) noexcept
{
    std::array<f_type, N> res;
    for (f_type i : vec) {
        res[i] = sigmoid(i);
    }
    return res;
}

template<size_t N>
constexpr std::array<NN::f_type, N> vector_add(const std::array<NN::f_type, N>& a1, const std::array<NN::f_type, N>& a2)
{
    std::array<NN::f_type, N> res;
    for (size_t i = 0; i < N; ++i) {
        res[i] = a1[i] + a2[i];
    }
    return res;
}

template<size_t N, size_t N2, size_t M2>
constexpr std::array<NN::f_type, M2> vector_to_mtx_dot(const std::array<NN::f_type, N>& mtx1, const array2D<N2, M2>& mtx2) noexcept
{
    static_assert(N == N2, "Dimensions should be the same!");
    std::array<NN::f_type, M2> res;
    for (size_t i = 0; i < M2; ++i) {
        NN::f_type cur_i = 0.;
        for (size_t j = 0; j < N; ++j) {
            cur_i += mtx1[j] * mtx2[j][i];
        }
        res[i] = cur_i;
    }

    return res;
}

template<size_t N1, size_t M1, size_t N2, size_t M2>
constexpr array2D<N1, M2> dot_mul(const array2D<N1, M1>& a1, const array2D<N2, M2>& a2) noexcept
{
    static_assert(M1 == N2, "Dimensions should be the same!");

    array2D<N1, M2> res;
    for (size_t i = 0; i < N1; ++i) {
        NN::f_type cur_val = 0.;
        for (size_t j = 0; j < M2; ++j) {
            for (size_t k = 0; k < M1; ++k) {
                cur_val += a1[i][k] * a2[k][j];
            }
            res[i][j] = cur_val;
        }
    }
    return res;
}
} // namespace NN