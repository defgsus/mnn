/**	@file

	@brief Serial Stack header

	@author def.gsus-
	@version 2012/10/19 started
*/
#ifndef MNN_STACK_SERIAL_H_INCLUDED
#define MNN_STACK_SERIAL_H_INCLUDED

#include <cmath>
#include <vector>
#include <iostream>

#include <mnn/layer.h>
#include <mnn/function.h>

namespace MNN {

template <typename Float>
class StackSerial : public Layer<Float>
{
	public:

	StackSerial();

	virtual ~StackSerial();

	// ----------- nn interface --------------

	/** set input and output size */
	virtual void resize(size_t nrIn, size_t nrOut);

	/** return size of input */
	virtual size_t nrIn() const;

	/** return size of output */
	virtual size_t nrOut() const;

	/** clear / randomize weights */
	virtual void brainwash();

	// ------- layer interface ---------------

	/** return nr of layers */
	virtual size_t nrLayer() const;

	/** add a new layer. ownership is taken */
	virtual void add(Layer<Float> * layer);

	// ------- propagation -------------------

	virtual void fprop(Float * input, Float * output);

	virtual void bprop(Float * error, Float * error_output = 0, Float global_learn_rate = 1);

	// ------- info --------------------------

	virtual const char * name() const { return "StackSerial"; }

	virtual void info(std::ostream &out = std::cout) const;

	virtual void dump(std::ostream &out = std::cout) const;


	protected:

	void resizeBuffers_();

	/** intermediate buffers */
	std::vector<std::vector<Float> > buffer_;

	/** inidividual layers */
	std::vector<Layer<Float>*> layer_;
};

#include <mnn/stack_serial_impl.inl>

} // namespace MNN

#endif // MNN_STACK_SERIAL_H_INCLUDED
