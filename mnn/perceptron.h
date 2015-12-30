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

namespace MNN {

template <typename Float, class ActFunc>
class Perceptron : public Layer<Float>
{
	public:

    Perceptron(size_t numIn, size_t numOut, Float learnRate = 1, bool biasCell = true);

	virtual ~Perceptron();

    Float momentum() const { return momentum_; }
    void setMomentum(Float m) { momentum_ = m; }

	// ----------- nn interface --------------

    virtual void resize(size_t numIn, size_t numOut) override;
    virtual void grow(size_t nrIn, size_t nrOut, Float randomDev) override;
    virtual void brainwash() override;

    virtual size_t numIn() const override;
    virtual size_t numOut() const override;
    virtual const Float* inputs() const override { return &input_[0]; }
    virtual const Float* outputs() const override { return &output_[0]; }
    virtual const Float* weights() const override { return &weight_[0]; }

    virtual Float weight(size_t input, size_t output) const override
        { return weights()[output * input_.size() + input]; }

	// ------- propagation -------------------

    virtual void fprop(const Float * input, Float * output) override;

    virtual void bprop(const Float * error, Float * error_output = 0,
                       Float global_learn_rate = 1) override;

	// ------- info --------------------------

    virtual const char * id() const override { return "Perceptron"; }
    virtual const char * name() const override { return "Perceptron"; }
    virtual void info(std::ostream &out = std::cout) const override;
    virtual void dump(std::ostream &out = std::cout) const override;

    // ------------- io ---------------

    virtual void serialize(std::ostream&) const override;
    virtual void deserialize(std::istream&) override;

protected:

	std::vector<Float>
		input_,
		output_,
        weight_,
        prevDelta_;

    Float learnRate_,
          momentum_;

    bool biasCell_;
};

#include "perceptron_impl.inl"

} // namespace MNN

#endif // MNN_PERCEPTRON_H_INCLUDED
