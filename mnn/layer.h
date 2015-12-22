/**	@file

	@brief Layer base abstract

	@author def.gsus-
	@version 2012/10/15 started
*/
#ifndef MNN_LAYER_H_INCLUDED
#define MNN_LAYER_H_INCLUDED

#include <iostream>
#include <fstream>

#include "function.h"
#include "exception.h"

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

	/** clear / randomize weights */
	virtual void brainwash() = 0;

    // -------- data access ---------------

    /** return size of input */
    virtual size_t numIn() const = 0;

    /** return size of output */
    virtual size_t numOut() const = 0;

    /** Returns pointer to continous input states */
    virtual const Float* input() const = 0;
    /** Returns pointer to continous output states */
    virtual const Float* output() const = 0;

    Float input(size_t index) const { return input()[index]; }
    Float output(size_t index) const { return output()[index]; }

	// ---- propagation -------

	/** forward propagate */
    virtual void fprop(const Float * input, Float * output) = 0;

	/** backward propagate the error derivative, and adjust weights */
    virtual void bprop(const Float * error, Float * error_output = 0,
                       Float global_learn_rate = 1) = 0;

	// -------- info ----------

    /** Return a persistent identifier for the layer type */
    virtual const char * id() const = 0;

    /** Return a nice name for the layer type */
    virtual const char * name() const = 0;

	/** print an overview of the network */
	virtual void info(std::ostream &out = std::cout) const = 0;

	/** print (complete) internal data */
	virtual void dump(std::ostream &out = std::cout) const = 0;

    // ------------- io ---------------

    /** Saves the layer to a file using serialize(std::ostream&).
        @throws MNN::Exception */
    void saveAscii(const std::string& filename) const;

    /** Loads the layer from a file using deserialize(std::ostream&).
        @throws MNN::Exception */
    void loadAscii(const std::string& filename);

    /** Serializes the layer to an ASCII stream. */
    virtual void serialize(std::ostream&) const = 0;

    /** Deserializes the layer from an ASCII stream.
        @throws MNN::Exception on error */
    virtual void deserialize(std::istream&) = 0;

	// ------ protected space ----------------

	protected:

    /**disable copy*/ Layer& operator=(const Layer&) = delete;
    /**disable copy*/ Layer(const Layer&) = delete;

};

// ------------------------ impl ---------------------------

template <typename Float>
void Layer<Float>::saveAscii(const std::string& filename) const
{
    std::fstream fs;
    fs.open(filename, std::ios_base::out);
    if (!fs.is_open())
        MNN_EXCEPTION("Could not open file for writing '" << filename << "'");
    serialize(fs);
    fs.close();
}


template <typename Float>
void Layer<Float>::loadAscii(const std::string& filename)
{
    std::fstream fs;
    fs.open(filename, std::ios_base::in);
    if (!fs.is_open())
        MNN_EXCEPTION("Could not open file for reading '" << filename << "'");
    deserialize(fs);
    fs.close();
}

} // namespace MNN

#endif // MNN_LAYER_H_INCLUDED
