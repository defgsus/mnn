/** @file

    @brief

    <p>(c) 2016, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 1/11/2016</p>
*/

#ifndef MNN_CONVOLUTION_H
#define MNN_CONVOLUTION_H

#include <cmath>
#include <cassert>
#include <vector>
#include <iostream>

#include "layer.h"
#include "interface.h"

namespace MNN {

/** 2D Convolution.

    @note multiple feature maps are NOT implemented currently
*/
template <typename Float, class ActFunc>
class Convolution
        : public Layer<Float>
        , public GetMomentumInterface<Float>
        , public SetMomentumInterface<Float>
{
    public:

    Convolution(size_t inputWidth, size_t inputHeight,
                size_t kernelWidth, size_t kernelHeight,
                Float learnRate = 1);

    Convolution(size_t inputWidth, size_t inputHeight, size_t inputMaps,
                size_t kernelWidth, size_t kernelHeight, size_t outputMaps,
                Float learnRate = 1);

    virtual ~Convolution();

    // ----------- copying -------------------

    virtual Convolution<Float, ActFunc> * cloneClass() const override
        { return new Convolution<Float, ActFunc>(
                    inputWidth(), inputHeight(), kernelWidth(), kernelHeight(), learnRate_); }

    virtual Convolution<Float, ActFunc>& operator = (const Layer<Float>&) override;

    // --------- MomentumInterface -----------

    virtual Float momentum() const override { return momentum_; }
    virtual void setMomentum(Float m) override { momentum_ = m; }

    // ----------- nn interface --------------

    virtual void resize(size_t inputWidth, size_t inputHeight,
                        size_t kernelWidth, size_t kernelHeight);
    virtual void resize(size_t inputWidth, size_t inputHeight, size_t inputMaps,
                        size_t kernelWidth, size_t kernelHeight, size_t outputMaps);

    virtual void resize(size_t numIn, size_t numOut) override
        { assert(!"Can't use this resize function"); (void)numIn; (void)numOut; }
    virtual void grow(size_t nrIn, size_t nrOut, Float randomDev) override
        { assert(!"Can't use the grow function"); (void)nrIn; (void)nrOut; (void)randomDev; }
    virtual void brainwash(Float variance = 1.) override;

    virtual size_t inputWidth() const { return inputWidth_; }
    virtual size_t inputHeight() const { return inputHeight_; }
    virtual size_t kernelWidth() const { return kernelWidth_; }
    virtual size_t kernelHeight() const { return kernelHeight_; }
    virtual size_t scanWidth() const { return scanWidth_; }
    virtual size_t scanHeight() const { return scanHeight_; }

    virtual size_t numIn() const override;
    virtual size_t numOut() const override;
    virtual const Float* inputs() const override { return &input_[0]; }
    virtual const Float* outputs() const override { return &output_[0]; }
    virtual const Float* weights() const override { return &weight_[0]; }
    virtual Float* weights() override { return &weight_[0]; }

    virtual Float weight(size_t input, size_t output) const override
        { return weights()[output * input_.size() + input]; }
    virtual void setWeight(size_t input, size_t output, Float w) override
        { weights()[output * input_.size() + input] = w; }

    // ------- propagation -------------------

    virtual void fprop(const Float * input, Float * output) override;

    virtual void bprop(const Float * error, Float * error_output = 0,
                       Float global_learn_rate = 1) override;

    // ------- info --------------------------

    virtual const char * id() const override { return "Convolution"; }
    virtual const char * name() const override { return "Convolution"; }
    virtual size_t numParameters() const override { return weight_.size(); }
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
        output_,
        weight_,
        prevDelta_;

    size_t
        inputMaps_, outputMaps_,
        inputWidth_, inputHeight_,
        kernelWidth_, kernelHeight_,
        scanWidth_, scanHeight_;
    Float
        learnRate_,
        momentum_;
};

#include "convolution_impl.inl"

} // namespace MNN

#endif // MNN_CONVOLUTION_H

