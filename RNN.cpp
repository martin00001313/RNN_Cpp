#include <cmath>

#include "./RNN.h"

constexpr NN::f_type RNN::get_output(const std::array<NN::f_type, RNN::seq_length>& entity) const noexcept
{
    std::array<NN::f_type, seq_length> cur{0};
    std::array<NN::f_type, hidden_dim> prev_state{0};

    NN::f_type res = 0.;
    for (size_t i = 0; i < seq_length ; ++i) {
        cur[i] = entity[i];
        if (i != 0) {
            cur[i-1] = 0.;
        }
        const auto dot_input = NN::vector_to_mtx_dot(cur, input_layer);
        const auto dot_hidden = NN::vector_to_mtx_dot(prev_state, hidden_layer);
        const auto sum_vec = NN::vector_add(dot_hidden, dot_input);
        const auto sg = NN::sigmoid(sum_vec);
        res = NN::vector_to_mtx_dot(sg, output_layer)[0];
        prev_state = sg;
    }
    return res;
}

constexpr NN::f_type RNN::get_loss(const NN::sample_type<RNN::seq_length>& entry) const noexcept
{
    return entry.second - get_output(entry.first);
}

constexpr void RNN::train_network(train_data_type& data, const validation_data_type& validation_data) noexcept
{
    std::array<std::pair<std::array<NN::f_type, hidden_dim>, std::array<NN::f_type, hidden_dim>>, seq_length> layers;

    for (auto& i : data) {
        auto& cur_input = i.first;
        const auto& cur_y = i.second;

        NN::f_type n_res = 0.;
        std::array<NN::f_type, hidden_dim> prev_s = {0.};

        // Forward pass
        for (size_t i = 0; i < seq_length; ++i) {
            std::remove_const_t<std::remove_reference_t<decltype(cur_input)>> tmp_data = {0.};
            tmp_data[i] = cur_input[i];
            const auto dot_input = NN::vector_to_mtx_dot(tmp_data, input_layer);
            const auto dot_hidden = NN::vector_to_mtx_dot(prev_s, hidden_layer);
            const auto sum_vec = NN::vector_add(dot_hidden, dot_input);
            const auto sg = NN::sigmoid(sum_vec);
            n_res = NN::vector_to_mtx_dot(sg, output_layer)[0];
            layers[i].first = sg;
            layers[i].second = prev_s;
        }
        const NN::f_type loss = (n_res - i.second);

        // Backword pass
        decltype(input_layer) du = {0.};
        decltype(hidden_layer) dw = {0.};
        decltype(output_layer) dv = {0.};

        decltype(input_layer) du_t = {0.};
        decltype(hidden_layer) dw_t = {0.};
        decltype(output_layer) dv_t = {0.};

        decltype(input_layer) du_i = {0.};
        decltype(hidden_layer) dw_i = {0.};
        
        for (size_t t = 0; t < seq_length; ++t) {
            
        }
    }   
}
