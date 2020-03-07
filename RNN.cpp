#include <cmath>

#include "./RNN.h"

NN::f_type RNN::get_output(const std::array<NN::f_type, RNN::seq_length>& entity) noexcept
{
    std::array<NN::f_type, seq_length> cur{0};
    std::array<NN::f_type, hidden_dim> prev_state{0};

    NN::f_type res;
    for (size_t i = 0; i < seq_length ; ++i) {
        cur[i] = entity[i];
        if (i != 0) {
            cur[i-1] = 0.;
        }
        const auto input_res = execute_input_layer(cur);
        const auto cur_hidden = execute_hidden_layer(input_res);
        const auto dot_input = NN::vector_to_mtx_dot(cur, input_layer);
        const auto dot_hidden = NN::vector_to_mtx_dot(prev_state, hidden_layer);
        const auto sum_vec = NN::vector_add(dot_hidden, dot_input);
        const auto sg = NN::sigmoid(sum_vec);
        res = NN::vector_to_mtx_dot(sg, output_layer)[0];
        prev_state = sg;
    }
    return res;
}