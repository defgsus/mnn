/** @file

    @brief

    <p>(c) 2016, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 1/14/2016</p>
*/

#ifndef MNNSRC_INTERFACE_H
#define MNNSRC_INTERFACE_H

#include <cstddef> // for size_t

namespace MNN {



// --------------- dropout --------------------

enum DropOutMode
{
    /** No drop-out */
    DO_OFF,
    /** Drop-out during training.
        Hidden cells are disabled with a probability of .5 */
    DO_TRAIN,
    /** Drop-out during performance.
        A network trained with DO_TRAIN will half the output
        of the hidden cells */
    DO_PERFORM
};


/** Interface for setting the dropout mode and probability */
template <typename Float>
class SetDropOutInterface
{
public:

    /** Sets dropout mode */
    virtual void setDropOutMode(DropOutMode m) = 0;

    /** Sets dropout probability */
    virtual void setDropOut(Float probability) = 0;

};

/** Interface for getting the dropout mode and probability */
template <typename Float>
class GetDropOutInterface
{
public:

    /** Return dropout mode */
    virtual DropOutMode dropOutMode() const = 0;
    /** Return dropout probability */
    virtual Float dropOut() const = 0;
};



// ---------------- learnrate ----------------

/** Interface for setting learning rate */
template <typename Float>
class SetLearnRateInterface
{
public:

    virtual void setLearnRate(Float lr) = 0;
};

/** Interface for getting learning rate*/
template <typename Float>
class GetLearnRateInterface
{
public:

    virtual Float learnRate() const = 0;
};


// ---------------- bias learnrate ----------------

/** Interface for setting bias learning rate */
template <typename Float>
class SetLearnRateBiasInterface
{
public:

    virtual void setLearnRateBias(Float lr) = 0;
};

/** Interface for getting bias learning rate*/
template <typename Float>
class GetLearnRateBiasInterface
{
public:

    virtual Float learnRateBias() const = 0;
};


// ---------------- softmax ----------------

/** Interface for setting softmax mode */
class SetSoftmaxInterface
{
public:

    virtual void setSoftmax(bool enable) = 0;
};

/** Interface for getting softmax mode */
class GetSoftmaxInterface
{
public:

    virtual bool isSoftmax() const = 0;
};


// ---------------- softmax ----------------

/** Interface for setting use of bias cells */
class SetBiasEnabledInterface
{
public:

    virtual void setBiasEnabled(bool enable) = 0;
};

/** Interface for getting use of bias mode */
class GetBiasEnabledInterface
{
public:

    virtual bool isBiasEnabled() const = 0;
};



// ---------------- momentum ----------------

/** Interface for setting momentum rate */
template <typename Float>
class SetMomentumInterface
{
public:

    virtual void setMomentum(Float m) = 0;
};

/** Interface for getting momentum rate */
template <typename Float>
class GetMomentumInterface
{
public:

    virtual Float momentum() const = 0;
};



// ------------------- cd ------------------

/** Contrastive divergence training. */
template <typename Float>
class ContrastiveDivergenceInterface
{
public:
    /** Returns the average absolute reconstruction error. */
    virtual Float contrastiveDivergence(
            const Float* input, size_t numSteps = 1, Float learn_rate = 1) = 0;
};



// ------- reconstruction / auto-encoding ------

template <typename Float>
class ReconstructionInterface
{
public:

    /** Build a reconstruction from the input */
    virtual void reconstruct(const Float* input, Float* reconstruction) = 0;

    /** One training step to reconstruct the input.
        @returns the average absolute error */
    virtual Float reconstructionTraining(
            const Float* input, Float learn_rate = 1)
        { return reconstructionTraining(input, input, learn_rate); }

    /** One training step to reconstruct the input.
        For training a denoising auto-encoder.
        @p encoder_input will be encoded and decoded and compared
        with @p expected_input to calculate the construction error.
        @returns the average absolute error */
    virtual Float reconstructionTraining(
            const Float* encoder_input, const Float* expected_input,
            Float learn_rate = 1) = 0;
};



// ------------------ convolution ------------------------------

class ConvolutionInterface
{
public:
    virtual size_t inputWidth() const = 0;
    virtual size_t inputHeight() const = 0;
    virtual size_t kernelWidth() const = 0;
    virtual size_t kernelHeight() const = 0;
    virtual size_t scanWidth() const = 0;
    virtual size_t scanHeight() const = 0;
    virtual size_t strideX() const = 0;
    virtual size_t strideY() const = 0;
    virtual size_t numParallelMaps() const = 0;
    virtual size_t numInputMaps() const = 0;
    virtual size_t numOutputMaps() const = 0;

    virtual void resize(size_t inputWidth, size_t inputHeight,
                        size_t kernelWidth, size_t kernelHeight)
        { resize(inputWidth, inputHeight, 1, kernelWidth, kernelHeight, 1); }

    virtual void resize(size_t inputWidth, size_t inputHeight, size_t numInputMaps,
                        size_t kernelWidth, size_t kernelHeight, size_t numParallelMaps)
        { resize(inputWidth, inputHeight, numInputMaps, 1, 1,
                 kernelWidth, kernelHeight, numParallelMaps); }

    virtual void resize(size_t inputWidth, size_t inputHeight, size_t numInputMaps,
                        size_t strideX, size_t strideY,
                        size_t kernelWidth, size_t kernelHeight, size_t numParallelMaps) = 0;
};



// ------------------ non-member functions ---------------------

template <typename Float>
class Layer;

template <typename Float>
void setLearnRate(Layer<Float>* l, Float v)
{
    if (auto d = dynamic_cast<SetLearnRateInterface<Float>*>(l))
        d->setLearnRate(v);
}

template <typename Float>
void setLearnRateBias(Layer<Float>* l, Float v)
{
    if (auto d = dynamic_cast<SetLearnRateBiasInterface<Float>*>(l))
        d->setLearnRateBias(v);
}

template <typename Float>
void setMomentum(Layer<Float>* l, Float v)
{
    if (auto d = dynamic_cast<SetMomentumInterface<Float>*>(l))
        d->setMomentum(v);
}

template <typename Float>
void setSoftmax(Layer<Float>* l, bool enable)
{
    if (auto d = dynamic_cast<SetSoftmaxInterface*>(l))
        d->setSoftmax(enable);
}

template <typename Float>
void setBiasEnabled(Layer<Float>* l, bool enable)
{
    if (auto d = dynamic_cast<SetBiasEnabledInterface*>(l))
        d->setBiasEnabled(enable);
}

template <typename Float>
void setDropOutMode(Layer<Float>* l, DropOutMode m)
{
    if (auto d = dynamic_cast<SetDropOutInterface<Float>*>(l))
        d->setDropOutMode(m);
}

template <typename Float>
void setDropOut(Layer<Float>* l, Float prob)
{
    if (auto d = dynamic_cast<SetDropOutInterface<Float>*>(l))
        d->setDropOut(prob);
}

} // namespace MNN

#endif // MNNSRC_INTERFACE_H

