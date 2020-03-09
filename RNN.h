#include "./utils.h"

class RNN
{
public:
    constexpr static size_t N = 200;
    constexpr static size_t seq_length = 50;
    constexpr static size_t V_S = 50;
    constexpr static size_t hidden_dim = 100;
    constexpr static size_t output_dim = 1;

private:
    NN::array2D<seq_length, hidden_dim> input_layer;
    NN::array2D<hidden_dim, hidden_dim> hidden_layer;
    NN::array2D<hidden_dim, output_dim> output_layer;

    using validation_data_type = std::result_of_t<decltype(&NN::get_data<V_S, seq_length>)(size_t)>;
    using train_data_type = std::result_of_t<decltype(&NN::get_data<N, seq_length>)(size_t)>;

public:

    RNN() : input_layer(NN::get_weights<seq_length, hidden_dim>())
        , hidden_layer(NN::get_weights<hidden_dim, hidden_dim>())
        , output_layer(NN::get_weights<hidden_dim, output_dim>())
    {
    }

    constexpr NN::f_type get_output(const std::array<NN::f_type, seq_length>& entity) const noexcept;

    constexpr void train_network(train_data_type& data, const validation_data_type& validation_data) noexcept;

private:
    constexpr NN::f_type get_loss(const NN::sample_type<seq_length>&) const noexcept;
};