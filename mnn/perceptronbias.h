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
#include "interface.h"

namespace MNN {

/** Perceptron with bias cells.
    Not perfectly working yet */
template <typename Float, class ActFunc>
class PerceptronBias
        : public Layer<Float>
        , public GetMomentumInterface<Float>
        , public SetMomentumInterface<Float>
        , public ReconstructionInterface<Float>
{
    public:

    PerceptronBias(size_t numIn, size_t numOut, Float learnRate = 1);

    virtual ~PerceptronBias();

    // ----------- copying -------------------

    virtual PerceptronBias<Float, ActFunc> * cloneClass() const override
        { return new PerceptronBias<Float, ActFunc>(numIn(), numOut(), learnRate_); }

    virtual PerceptronBias<Float, ActFunc>& operator = (const Layer<Float>&) override;

    // --------- MomentumInterface -----------

    virtual Float momentum() const override { return momentum_; }
    virtual void setMomentum(Float m) override { momentum_ = m; }

    // ---

    Float learnRateBias() const { return learnRateBias_; }
    void setLearnRateBias(Float lr) { learnRateBias_ = lr; }

    // -------- ReconstructionInterface ------

    virtual void reconstruct(const Float* input, Float* reconstruction) override;
    using ReconstructionInterface<Float>::reconstructionTraining;
    virtual Float reconstructionTraining(
            const Float *decoder_input, const Float* expected_input,
            Float learn_rate = 1) override;

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

    virtual const Float* biases() const { return &bias_[0]; }
    virtual Float* biases() { return &bias_[0]; }

    // ------- propagation -------------------

    virtual void fprop(const Float * input, Float * output) override;

    virtual void bprop(const Float * error, Float * error_output = 0,
                       Float global_learn_rate = 1) override;

    // ------- info --------------------------

    virtual const char * id() const override { return "PerceptronBias"; }
    virtual const char * name() const override { return "PerceptronBias"; }
    virtual size_t numParameters() const override { return weight_.size() + bias_.size(); }
    virtual void info(std::ostream &out = std::cout,
                      const std::string& postFix = "") const override;
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
        prevDelta_,
        reconInput_,
        reconError_,
        reconOutput_;

    Float learnRate_,
          learnRateBias_,
          momentum_;
};

#include "perceptronbias_impl.inl"

} // namespace MNN

#endif // MNN_PERCEPTRONBIAS_H_INCLUDED

