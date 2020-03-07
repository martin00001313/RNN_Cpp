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

    std::result_of_t<decltype(&NN::get_data<N, seq_length>)(size_t)> train_data;
    std::result_of_t<decltype(&NN::get_data<V_S, seq_length>)(size_t)> validation_data;

public:

    RNN() : input_layer(NN::get_weights<seq_length, hidden_dim>())
        , hidden_layer(NN::get_weights<hidden_dim, hidden_dim>())
        , output_layer(NN::get_weights<hidden_dim, output_dim>())
        , train_data(NN::get_data<N, seq_length>(0))
        , validation_data(NN::get_data<V_S, seq_length>(N))
    {
    }

    NN::f_type get_output(const std::array<NN::f_type, seq_length>& entity) noexcept;

    NN::f_type get_loss() const noexcept;

private:
    std::array<NN::f_type, hidden_dim> execute_input_layer(const std::array<NN::f_type, seq_length>& entity) noexcept;
    std::array<NN::f_type, hidden_dim> execute_hidden_layer(const std::array<NN::f_type, hidden_dim>& entity) noexcept;
    NN::f_type execute_output_layer(const std::array<NN::f_type, hidden_dim>& entity) noexcept;
};