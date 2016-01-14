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
#include "interface.h"
#include "activation.h"

namespace MNN {

template <typename Float, class ActFunc = MNN::Activation::Logistic>
class Rbm
        : public Layer<Float>
        , public GetMomentumInterface<Float>
        , public SetMomentumInterface<Float>
        , public ContrastiveDivergenceInterface<Float>
{
    public:

    Rbm(size_t numIn, size_t numOut, Float learnRate = 1, bool haveBiasCell = false);

    virtual ~Rbm();

    // ----------- copying -------------------

    virtual Rbm<Float, ActFunc> * cloneClass() const override
        { return new Rbm<Float, ActFunc>(numIn(), numOut(), learnRate_, biasCell_); }

    virtual Rbm<Float, ActFunc>& operator = (const Layer<Float>&) override;

    // --------- MomentumInterface -----------

    virtual Float momentum() const override { return momentum_; }
    virtual void setMomentum(Float m) override { momentum_ = m; }

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

    virtual Float weight(size_t input, size_t output) const override
        { return weights()[output * input_.size() + input]; }
    virtual void setWeight(size_t input, size_t output, Float w) override
        { weights()[output * input_.size() + input] = w; }

    // ------- propagation -------------------

    virtual void fprop(const Float * input, Float * output) override;

    virtual void bprop(const Float * error, Float * error_output = 0,
                       Float global_learn_rate = 1) override;

    /** Contrastive divergence training.
        Returns the summed absolute reconstruction error */
    virtual Float contrastive_divergence(
            const Float* input, size_t numSteps = 1, Float learn_rate = 1) override;

    /** Returns the sum of the absolute difference between
        @p input and the current input state */
    Float compareInput(const Float* input) const;

    // ------- info --------------------------

    virtual const char * id() const override { return "RBM"; }
    virtual const char * name() const override { return "RBM"; }
    virtual size_t numParameters() const override { return weight_.size(); }
    virtual void info(std::ostream &out = std::cout,
                      const std::string& postFix = "") const override;
    virtual void dump(std::ostream &out = std::cout) const override;

    virtual Float getWeightAverage() const override;

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

    Float learnRate_,
          momentum_;

    bool biasCell_;
};

#include "rbm_impl.inl"

} // namespace MNN

#endif // MNN_RBM_H_INCLUDED

