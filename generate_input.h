/** @file

    @brief

    <p>(c) 2016, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 1/12/2016</p>
*/

#ifndef GENERATE_INPUT_H
#define GENERATE_INPUT_H

#include <cassert>
#include <vector>
#include "mnn/function.h"

template <typename Float>
class GenerateInput
{
public:

    GenerateInput()
        : error_    (-1.)
    { }

    const Float* input() const { return &inputBest_[0]; }
    const Float* output() const { return &output_[0]; }
    Float error() const { return error_; }
    Float errorWorst() const { return errorWorst_; }

    void setExpectedOutput(const Float* v, size_t num);

    void initializeInput();
    void mutateInput();

    template <class Net>
    void approximateInput(Net& net, size_t numSteps);


    std::vector<Float>
        input_, inputBest_,
        output_, outputExpect_;
    Float error_, errorWorst_;
};


template <typename Float>
void GenerateInput<Float>::setExpectedOutput(const Float *v, size_t num)
{
    output_.resize(num);
    outputExpect_.resize(num);
    for (auto& o : outputExpect_)
        o = *v++;
}

template <typename Float>
void GenerateInput<Float>::initializeInput()
{
    for (auto& v : input_)
        v = MNN::rnd(0.2, 0.21);
    error_ = -1.;
}

template <typename Float>
template <class Net>
void GenerateInput<Float>::approximateInput(Net& net, size_t numSteps)
{
    // resize/reinit if necessary
    if (input_.size() != net.numIn())
    {
        input_.resize(net.numIn());
        inputBest_.resize(input_.size());
        initializeInput();
    }

    assert(output_.size() == net.numOut() && outputExpect_.size() == net.numOut() );

    errorWorst_ = 0.;

    for (size_t step = 0; step < numSteps; ++step)
    {
        net.fprop(&input_[0], &output_[0]);

        // get error
        Float e = 0.;
        for (size_t i = 0; i < output_.size(); ++i)
            e += std::abs(outputExpect_[i] - output_[i]);
        e = e / output_.size() * 100.;

        errorWorst_ = std::max(errorWorst_, e);

        // made progress?
        if (error_ < 0. || e < error_)
        {
            error_ = e;
            inputBest_ = input_;
        }
        // otherwise reverse mutation
        else
            input_ = inputBest_;

        mutateInput();
    }
}

template <typename Float>
void GenerateInput<Float>::mutateInput()
{
    for (auto& i : input_)
        if (MNN::rnd(0., 1.) < 0.3)
            i = std::max(-.1, std::min(1.,
                    i + MNN::rnd(-1., 1.) * MNN::rnd(0., .002)
                                     ));
}

#endif // GENERATE_INPUT_H

