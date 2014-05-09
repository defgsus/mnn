/**	@file

	@brief Perceptron header

	@author def.gsus-
	@version 2012/10/15 started
*/
#ifndef MNN_PERCEPTRON_H_INCLUDED
#define MNN_PERCEPTRON_H_INCLUDED

#include <cmath>
#include <vector>
#include <iostream>

#include <mnn/layer.h>
#include <mnn/function.h>

namespace MNN {

template <typename Float, class ActFunc>
class Perceptron : public Layer<Float>
{
	public:

    Perceptron(size_t numIn, size_t numOut, Float learnRate = 1);

	virtual ~Perceptron();

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

	virtual void fprop(Float * input, Float * output);

	virtual void bprop(Float * error, Float * error_output = 0, Float global_learn_rate = 1);

	// ------- info --------------------------

	virtual const char * name() const { return "Perceptron"; }

	virtual void info(std::ostream &out = std::cout) const;

	virtual void dump(std::ostream &out = std::cout) const;

	protected:

	std::vector<Float>
		input_,
		output_,
		weight_;

	Float learnRate_;
};

#include <mnn/perceptron_impl.inl>

} // namespace MNN

#endif // MNN_PERCEPTRON_H_INCLUDED
