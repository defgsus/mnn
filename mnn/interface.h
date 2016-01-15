/** @file

    @brief

    <p>(c) 2016, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 1/14/2016</p>
*/

#ifndef MNN_INTERFACE_H
#define MNN_INTERFACE_H

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

} // namespace MNN

#endif // MNN_INTERFACE_H

