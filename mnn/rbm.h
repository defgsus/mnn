/** @file rbm.h

    @brief Restricted Boltzman Machine

    <p>(c) 2015, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 12/22/2015</p>
*/

#ifndef MNN_RBM_H_INCLUDED
#define MNN_RBM_H_INCLUDED

#include <cmath>
#include <vector>
#include <iostream>

#include "layer.h"
#include "activation.h"

namespace MNN {

template <typename Float, class ActFunc = MNN::Activation::Logistic>
class Rbm : public Layer<Float>
{
    public:

    Rbm(size_t numIn, size_t numOut, Float learnRate = 1, bool haveBiasCell = false);

    virtual ~Rbm();

    Float momentum() const { return momentum_; }
    void setMomentum(Float m) { momentum_ = m; }

    // ----------- nn interface --------------

    virtual void resize(size_t numIn, size_t numOut) override;
    virtual void brainwash() override;

    virtual size_t numIn() const override;
    virtual size_t numOut() const override;
    virtual const Float* inputs() const override { return &input_[0]; }
    virtual const Float* outputs() const override { return &output_[0]; }
    virtual const Float* weights() const override { return &weight_[0]; }

    virtual Float weight(size_t input, size_t output) const override
        { return weights()[output * input_.size() + input]; }

    // ------- propagation -------------------

    virtual void setDropOut(DropOutMode mode) override;

    virtual void fprop(const Float * input, Float * output) override;

    virtual void bprop(const Float * error, Float * error_output = 0,
                       Float global_learn_rate = 1) override;

    /** Contrastive divergence training.
        Returns the summed absolute reconstruction error */
    Float cd(const Float* input, size_t numSteps = 1, Float learn_rate = 1);

    /** Returns the sum of the absolute difference between
        @p input and the current input state */
    Float compareInput(const Float* input) const;

    // ------- info --------------------------

    virtual const char * id() const override { return "RBM"; }
    virtual const char * name() const override { return "RBM"; }
    virtual void info(std::ostream &out = std::cout) const override;
    virtual void dump(std::ostream &out = std::cout) const override;

    Float getWeightAverage() const;

    // ------------- io ---------------

    virtual void serialize(std::ostream&) const override;
    virtual void deserialize(std::istream&) override;

protected:

    void copyInput_(const Float* input);

    /** Propagates input_ to output_ through weightUp_ */
    void propUp_();

    /** Propagates output_ to input_ through weightDown_ */
    void propDown_();

    /** Makes states binary */
    static void makeBinary_(Float* states, size_t num);
    void makeBinaryInput_() { makeBinary_(&input_[0], input_.size()); }
    void makeBinaryOutput_() { makeBinary_(&output_[0], output_.size()); }

    void getCorrelation_(Float* matrix) const;

    /** Adjust weights by reconstruction/correlation error.
        Returns sum of errors */
    Float trainCorrelation_(Float learn_rate);

    std::vector<Float>
        input_,
        output_,
        weight_,
        prevDelta_,
        correlationData_,
        correlationModel_;
    std::vector<uint8_t>
        droppedCells_;

    Float learnRate_,
          momentum_;

    bool biasCell_;
    DropOutMode dropOutMode_;
};

#include "rbm_impl.inl"

} // namespace MNN

#endif // MNN_RBM_H_INCLUDED

