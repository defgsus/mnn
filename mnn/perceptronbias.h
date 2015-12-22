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

#include <mnn/layer.h>
#include <mnn/function.h>

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

    /** set input and output size */
    virtual void resize(size_t numIn, size_t numOut);

    /** return size of input */
    virtual size_t numIn() const;

    /** return size of output */
    virtual size_t numOut() const;

    /** clear / randomize weights */
    virtual void brainwash();

    // ------- propagation -------------------

    virtual void fprop(const Float * input, Float * output) override;

    virtual void bprop(const Float * error, Float * error_output = 0,
                       Float global_learn_rate = 1) override;

    // ------- info --------------------------

    virtual const char * name() const { return "PerceptronBias"; }

    virtual void info(std::ostream &out = std::cout) const;

    virtual void dump(std::ostream &out = std::cout) const;

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

#include <mnn/perceptronbias_impl.inl>

} // namespace MNN

#endif // MNN_PERCEPTRONBIAS_H_INCLUDED

