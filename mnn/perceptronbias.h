/** @file

    @brief

    <p>(c) 2015, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 12/21/2015</p>
*/

#ifndef MNN_PERCEPTRONBIAS_H_INCLUDED
#define MNN_PERCEPTRONBIAS_H_INCLUDED

#include <cmath>
#include <vector>
#include <iostream>

#include "layer.h"

namespace MNN {

/** Perceptron with bias cells.
    Not perfectly working yet */
template <typename Float, class ActFunc>
class PerceptronBias : public Layer<Float>
{
    public:

    PerceptronBias(size_t numIn, size_t numOut, Float learnRate = 1);

    virtual ~PerceptronBias();

    Float learnRateBias() const { return learnRateBias_; }
    void setLearnRateBias(Float lr) { learnRateBias_ = lr; }

    Float momentum() const { return momentum_; }
    void setMomentum(Float m) { momentum_ = m; }

    // ----------- nn interface --------------

    virtual void resize(size_t numIn, size_t numOut) override;
    virtual void grow(size_t nrIn, size_t nrOut, Float randomDev) override;
    virtual void brainwash(Float variance = 1.) override;

    virtual size_t numIn() const override;
    virtual size_t numOut() const override;
    virtual const Float* inputs() const override { return &input_[0]; }
    virtual const Float* outputs() const override { return &output_[0]; }
    virtual const Float* weights() const override { return &weight_[0]; }
    virtual Float* weights() override { return &weight_[0]; }

    // ------- propagation -------------------

    virtual void fprop(const Float * input, Float * output) override;

    virtual void bprop(const Float * error, Float * error_output = 0,
                       Float global_learn_rate = 1) override;

    // ------- info --------------------------

    virtual const char * id() const override { return "PerceptronBias"; }
    virtual const char * name() const override { return "PerceptronBias"; }
    virtual size_t numParameters() const override { return weight_.size() + bias_.size(); }
    virtual void info(std::ostream &out = std::cout) const override;
    virtual void dump(std::ostream &out = std::cout) const override;

    virtual Float getWeightAverage() const override;

    // ------------- io ---------------

    virtual void serialize(std::ostream&) const override;
    virtual void deserialize(std::istream&) override;

protected:

    std::vector<Float>
        input_,
        bias_,
        output_,
        weight_,
        prevDelta_;

    Float learnRate_,
          learnRateBias_,
          momentum_;
};

#include "perceptronbias_impl.inl"

} // namespace MNN

#endif // MNN_PERCEPTRONBIAS_H_INCLUDED

