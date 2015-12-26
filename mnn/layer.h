/**	@file layer.h

	@brief Layer base abstract

	@author def.gsus-
	@version 2012/10/15 started
    @version 2015/12/21 started major revision
*/
#ifndef MNN_LAYER_H_INCLUDED
#define MNN_LAYER_H_INCLUDED

#include <iostream>
#include <fstream>

#include "function.h"
#include "exception.h"

namespace MNN {


/** XXX Not used yet */
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



/** NN-Layer base class (abstract).

    <p>A layer is at it's basic level a set of inputs and outputs, which are
	in turn arrays of floats of probably different lengths.</p>

    <p>Internally, layers can be nodes, delaybuffers or whatever, or simply some
	function of the input, in which case a Layer isn't even a layer in the
	neuronal network sense.</p>

    <p>There are a few must-overrides, namely:
    @li resize(), which should set or reset both lengths, of input and output arrays.
    @li nrIn() and nrOut(), to return the lengths.
    @li input() and output(), to return pointers to the arrays
    @li brainwash(), to reset the weights, coefficients or whathaveyou.
    @li fprop(), this should propagte an array of Float from the input to the output.
    @li bprop(), this should at least propagate the output back to the input.
        as with back-propagation kind of nets, this should use the error
        and adjust the internal weights, while passing the derivative through.
    @li id() and name(), to help identify derived classes
    @li info(), to print a human-readable overview over the settings.
    @li dump(), to print (possibly much) internal data to some std::ostream.
    @li serialize() and deserialize(), to save and load from ascii data
	</p>

    <p>Each derived class should be copyable
    with operator= and the copy constructor</p>

    <p>The (de)serialization from/to text is choosen so that
    different Float template parameter or endianess do not
    cause problems for file persistence.</p>

 */
template <typename Float>
class Layer
{
	public:

    // ----------- ctor ----------------------

	Layer() { }
	virtual ~Layer() { }

	// ----------- nn interface --------------

    /** Set input and output size */
	virtual void resize(size_t nrIn, size_t nrOut) = 0;

    /** Clear states / randomize weights */
	virtual void brainwash() = 0;

    // -------- data access ---------------

    /** Return size of input */
    virtual size_t numIn() const = 0;

    /** Return size of output */
    virtual size_t numOut() const = 0;

    /** Return pointer to continous input values */
    virtual const Float* inputs() const = 0;
    /** Return pointer to continous output values */
    virtual const Float* outputs() const = 0;
    /** Return pointer to continous weight values */
    virtual const Float* weights() const = 0;

    /** Wrapper around inputs() */
    virtual Float input(size_t index) const { return inputs()[index]; }
    /** Wrapper around outputs() */
    virtual Float output(size_t index) const { return outputs()[index]; }
    /** Wrapper around weights(), returns weight between given input and output cell */
    virtual Float weight(size_t input, size_t output) const
        { return weights()[output * numIn() + input]; }

    // ---- propagation -------

    /** Forward propagate.
        Transmit the data in @p input to @p output. */
    virtual void fprop(const Float * input, Float * output) = 0;

    /** Backward propagate the error derivative, and adjust weights.
        Transmit data in @p error to @p error_output, if not NULL.
        Perform weight update if @p global_learn_rate != 0.
        The given learn rate is multiplied with any learnrate that might
        be set internally. */
    virtual void bprop(const Float * error, Float * error_output = 0,
                       Float global_learn_rate = 1) = 0;

	// -------- info ----------

    /** Return a persistent identifier for the layer type */
    virtual const char * id() const = 0;

    /** Return a nice name for the layer type */
    virtual const char * name() const = 0;

    /** Print an overview of the network */
	virtual void info(std::ostream &out = std::cout) const = 0;

    /** Print (complete) internal data */
	virtual void dump(std::ostream &out = std::cout) const = 0;

    // ------------- io ---------------

    /** Saves the layer to a file using serialize(std::ostream&).
        @throws MNN::Exception */
    void saveTextFile(const std::string& filename) const;

    /** Loads the layer from a file using deserialize(std::ostream&).
        @throws MNN::Exception */
    void loadTextFile(const std::string& filename);

    /** Serialize the layer to an ASCII stream.
        The serialized data should be everything that is needed to
        recreate the layer, e.g. number of inputs/outputs, learnrate,
        momentum, etc. and the weights. */
    virtual void serialize(std::ostream&) const = 0;

    /** Deserialize the layer from an ASCII stream.
        @throws MNN::Exception on error */
    virtual void deserialize(std::istream&) = 0;

};

// ------------------------ impl ---------------------------

template <typename Float>
void Layer<Float>::saveTextFile(const std::string& filename) const
{
    std::fstream fs;
    fs.open(filename, std::ios_base::out);
    if (!fs.is_open())
        MNN_EXCEPTION("Could not open file for writing '" << filename << "'");
    serialize(fs);
    fs.close();
}


template <typename Float>
void Layer<Float>::loadTextFile(const std::string& filename)
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
