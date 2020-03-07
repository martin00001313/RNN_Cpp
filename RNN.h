#include "./utils.h"

class RNN
{
private:
    constexpr static size_t N = 200;
    constexpr static size_t seq_length = 50;
    constexpr static size_t V_S = 50;
    constexpr static size_t hidden_dim = 100;
    constexpr static size_t output_dim = 1;

    NN::array2D<seq_length, hidden_dim> input_layer;
    NN::array2D<hidden_dim, hidden_dim> hidden_layer;
    NN::array2D<hidden_dim, output_dim> output_layer;

    std::pair<NN::sample_type<N - seq_length - V_S>, NN::sample_type<V_S>> train_validation_data;

public:

    RNN() : input_layer(NN::getWeights<seq_length, hidden_dim>())
        , hidden_layer(NN::getWeights<hidden_dim, hidden_dim>())
        , output_layer(NN::getWeights<hidden_dim, output_dim>())
    {
    }

    NN::f_type get_loss() const noexcept;
};