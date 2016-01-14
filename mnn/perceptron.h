/**	@file perceptron.h

	@brief Perceptron header

	@author def.gsus-
	@version 2012/10/15 started
*/
#ifndef MNN_PERCEPTRON_H_INCLUDED
#define MNN_PERCEPTRON_H_INCLUDED

#include <cmath>
#include <vector>
#include <iostream>

#include "layer.h"
#include "interface.h"

namespace MNN {

template <typename Float, class ActFunc>
class Perceptron
        : public Layer<Float>
        , public GetMomentumInterface<Float>
        , public SetMomentumInterface<Float>
        , public GetDropOutInterface<Float>
        , public SetDropOutInterface<Float>
{
	public:

    Perceptron(size_t numIn, size_t numOut, Float learnRate = 1, bool biasCell = true);

	virtual ~Perceptron();

    // ----------- copying -------------------

    virtual Perceptron<Float, ActFunc> * cloneClass() const override
        { return new Perceptron<Float, ActFunc>(numIn(), numOut(), learnRate_, biasCell_); }

    virtual Perceptron<Float, ActFunc>& operator = (const Layer<Float>&) override;

    // --------- MomentumInterface -----------

    virtual Float momentum() const override { return momentum_; }
    virtual void setMomentum(Float m) override { momentum_ = m; }

    // --------- DropOutInterface ------------

    virtual DropOutMode dropOutMode() const override { return dropOutMode_; }
    virtual void setDropOutMode(DropOutMode m) override { dropOutMode_ = m; }

    virtual Float dropOut() const override { return dropOut_; }
    virtual void setDropOut(Float probability) override { dropOut_ = probability; }

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

	// ------- info --------------------------

    virtual const char * id() const override { return "Perceptron"; }
    virtual const char * name() const override { return "Perceptron"; }
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
    std::vector<uint8_t>
        drop_input_;

    Float learnRate_,
          momentum_,
          dropOut_;

    DropOutMode dropOutMode_;

    bool biasCell_;
};

#include "perceptron_impl.inl"

} // namespace MNN

#endif // MNN_PERCEPTRON_H_INCLUDED
