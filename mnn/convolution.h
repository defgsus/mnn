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

    Convolves a 2D input with a trainable kernel.
    Supports multiple "feature maps" which are parallel convolving kernels.

    The size of the output of this layer is scanWidth() * scanHeight() * numOutputMaps(),
    scanWidth() beeing equal to (inputWidth() - kernelWidth()) / strideX() + 1.

    The number of output maps is equal to numInputMaps() * numParallelMaps(),
    meaning each input map (from a previous convolution stage)
    is convolved times numParallelMaps().

    There are numOutputMaps() kernels of size kernelWidth() * kernelHeight().
*/
template <typename Float, class ActFunc>
class Convolution
        : public Layer<Float>
        , public GetMomentumInterface<Float>
        , public SetMomentumInterface<Float>
        , public ConvolutionInterface
{
    public:

    Convolution(size_t inputWidth, size_t inputHeight,
                size_t kernelWidth, size_t kernelHeight,
                Float learnRate = 1);

    Convolution(size_t inputWidth, size_t inputHeight, size_t numInputMaps,
                size_t kernelWidth, size_t kernelHeight, size_t numParallelMaps,
                Float learnRate = 1);

    Convolution(size_t inputWidth, size_t inputHeight, size_t numInputMaps,
                size_t strideX, size_t strideY,
                size_t kernelWidth, size_t kernelHeight, size_t numParallelMaps,
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

    virtual void resize(size_t numIn, size_t numOut) override
        { assert(!"Can't use this resize function"); (void)numIn; (void)numOut; }
    virtual void grow(size_t nrIn, size_t nrOut, Float randomDev) override
        { assert(!"Can't use the grow function"); (void)nrIn; (void)nrOut; (void)randomDev; }
    virtual void brainwash(Float variance = 1.) override;

    // -------- ConvolutionInterface ----------

    using ConvolutionInterface::resize;
    virtual void resize(size_t inputWidth, size_t inputHeight, size_t numInputMaps,
                        size_t strideX, size_t strideY,
                        size_t kernelWidth, size_t kernelHeight, size_t numParallelMaps)
                                                                                override;

    virtual size_t inputWidth() const override { return inputWidth_; }
    virtual size_t inputHeight() const override { return inputHeight_; }
    virtual size_t kernelWidth() const override { return kernelWidth_; }
    virtual size_t kernelHeight() const override { return kernelHeight_; }
    virtual size_t scanWidth() const override { return scanWidth_; }
    virtual size_t scanHeight() const override { return scanHeight_; }
    virtual size_t strideX() const override { return strideX_; }
    virtual size_t strideY() const override { return strideY_; }
    virtual size_t numInputMaps() const override { return inputMaps_; }
    virtual size_t numParallelMaps() const override { return parallelMaps_; }
    virtual size_t numOutputMaps() const override { return inputMaps_ * parallelMaps_; }


    virtual size_t numIn() const override { return input_.size(); }
    virtual size_t numOut() const override { return output_.size(); }
    virtual const Float* inputs() const override { return &input_[0]; }
    virtual const Float* outputs() const override { return &output_[0]; }
    virtual const Float* weights() const override { return &weight_[0]; }
    virtual Float* weights() override { return &weight_[0]; }

    virtual Float weight(size_t input, size_t output) const override
        { assert(!"Can't use this function in Convolution"); (void)input; (void)output; }
    virtual void setWeight(size_t input, size_t output, Float w) override
        { assert(!"Can't use this function in Convolution"); (void)input; (void)output; (void)w; }

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
        outputErr_,
        prevDelta_,
        weightBuffer_;

    size_t
        inputMaps_, parallelMaps_,
        inputWidth_, inputHeight_,
        kernelWidth_, kernelHeight_,
        scanWidth_, scanHeight_,
        strideX_, strideY_;
    Float
        learnRate_,
        momentum_;
};

#include "convolution_impl.inl"

} // namespace MNN

#endif // MNN_CONVOLUTION_H

