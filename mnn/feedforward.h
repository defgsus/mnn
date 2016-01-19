/** @file

    @brief

    <p>(c) 2016, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 1/19/2016</p>
*/

#ifndef MNNSRC_DENSEMATRIX_H
#define MNNSRC_DENSEMATRIX_H

#include <cmath>
#include <vector>
#include <iostream>

#include "layer.h"
#include "interface.h"

namespace MNN {

/** Dense matrix layer, e.g. a perceptron or classic auto-encoder. */
template <typename Float, class ActFunc>
class FeedForward
        : public Layer<Float>
        , public GetLearnRateInterface<Float>
        , public SetLearnRateInterface<Float>
        , public GetLearnRateBiasInterface<Float>
        , public SetLearnRateBiasInterface<Float>
        , public GetBiasEnabledInterface
        , public SetBiasEnabledInterface
        , public GetMomentumInterface<Float>
        , public SetMomentumInterface<Float>
        , public GetSoftmaxInterface
        , public SetSoftmaxInterface
        , public ReconstructionInterface<Float>
{
    public:

    FeedForward(size_t numIn, size_t numOut, Float learnRate = 1, bool doBias = true);

    virtual ~FeedForward();

    // ----------- copying -------------------

    virtual FeedForward<Float, ActFunc> * cloneClass() const override
        { return new FeedForward<Float, ActFunc>(numIn(), numOut(), learnRate_, doBias_); }

    virtual FeedForward<Float, ActFunc>& operator = (const Layer<Float>&) override;

    // --------- LearnRateInterface ----------

    virtual Float learnRate() const override { return learnRate_; }
    virtual void setLearnRate(Float lr) override { learnRate_ = lr; }

    // --------- LearnRateBiasInterface ------

    virtual Float learnRateBias() const override { return learnRateBias_; }
    virtual void setLearnRateBias(Float lr) override { learnRateBias_ = lr; }

    // ----------- BiasEnabledInterface ------

    virtual void setBiasEnabled(bool enable) override { doBias_ = enable; }
    virtual bool isBiasEnabled() const override { return doBias_; }

    // --------- MomentumInterface -----------

    virtual Float momentum() const override { return momentum_; }
    virtual void setMomentum(Float m) override { momentum_ = m; }

    // --------- SoftmaxInterface ------------

    virtual void setSoftmax(bool enable) override { doSoftmax_ = enable; }
    virtual bool isSoftmax() const override { return doSoftmax_; }

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

    virtual size_t numIn() const override { return input_.size(); }
    virtual size_t numOut() const override { return output_.size(); }
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

    static const char* static_id() { return "feed_forward"; }
    virtual const char * id() const override { return static_id(); }
    virtual const char * name() const override { return "FeedForward"; }
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
        errorDer_,
        // scratch space for reconstruction
        reconInput_,
        reconError_,
        reconOutput_;

    Float learnRate_,
          learnRateBias_,
          momentum_;

    bool doBias_,
         doSoftmax_;
};

#include "feedforward_impl.inl"

} // namespace MNN

#endif // MNNSRC_DENSEMATRIX_H

