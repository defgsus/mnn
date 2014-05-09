/**	@file

	@brief Layer base abstract

	@author def.gsus-
	@version 2012/10/15 started
*/
#ifndef MNN_LAYER_H_INCLUDED
#define MNN_LAYER_H_INCLUDED

#include <iostream>

namespace MNN {

/** NN-Layer base class (abstract).

	<p>a layer is at it's basic level a set of inputs and outputs, which are
	in turn arrays of floats of probably different lengths.</p>

	<p>internally, layers can be nodes, delaybuffers or whatever, or simply some
	function of the input, in which case a Layer isn't even a layer in the
	neuronal network sense.</p>

    <p>there are a few must-overrides, namely:
    @li resize(), which should set or reset both lengths, of input and output arrays.
    @li nrIn() and nrOut(), to return the lengths.
    @li brainwash(), to reset the weights, coefficients or whathaveyou.
    @li fprop(), this should propagte an array of Float from the input to the output.
    @li bprop(), this should at least propagate the output back to the input.
        as with back-propagation kind of nets, this should use the error
        and adjust the internal weights, while passing the derivative through.
	</p>

	<p>optionally, Layer defines these (initially empty) functions: <br>
	- info(), to print a human-readable overview over the internal data.
	- dump(), to print (possibly much) internal data to some std::ostream.
    - energy(), to return a float representing the error or free energy of the system.

 */
template <typename Float>
class Layer
{
	public:

	Layer() { }
	virtual ~Layer() { }

	// ----------- nn interface --------------

	/** set input and output size */
	virtual void resize(size_t nrIn, size_t nrOut) = 0;

	/** return size of input */
    virtual size_t numIn() const = 0;

	/** return size of output */
    virtual size_t numOut() const = 0;

	/** clear / randomize weights */
	virtual void brainwash() = 0;

	// ---- propagation -------

	/** forward propagate */
	virtual void fprop(Float * input, Float * output) = 0;

	/** backward propagate the error derivative, and adjust weights */
	virtual void bprop(Float * error, Float * error_output = 0, Float global_learn_rate = 1) = 0;

	// -------- info ----------

	virtual const char * name() const = 0;

	/** should return error or free energy of the underlying system */
	//virtual Float energy() { return 0.0; }

	/** print an overview of the network */
	virtual void info(std::ostream &out = std::cout) const = 0;

	/** print (complete) internal data */
	virtual void dump(std::ostream &out = std::cout) const = 0;

	// ------ protected space ----------------

	protected:

	/**disable copy*/ Layer& operator=(const Layer&);
	/**disable copy*/ Layer(const Layer&);

};

} // namespace MNN

#endif // MNN_LAYER_H_INCLUDED
