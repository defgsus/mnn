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

#include "layer.h"

namespace MNN {

template <typename Float>
class StackSerial : public Layer<Float>
{
	public:

	StackSerial();

	virtual ~StackSerial();

	// ----------- nn interface --------------

    virtual void resize(size_t numIn, size_t numOut) override;
    virtual void grow(size_t nrIn, size_t nrOut, Float randomDev) override;
    virtual void brainwash(Float variance = 1.) override;

    virtual size_t numIn() const override;
    virtual size_t numOut() const override;
    virtual const Float* inputs() const override
        { return layer_.front()->inputs(); }
    virtual const Float* outputs() const override
        { return layer_.back()->outputs(); }
    virtual const Float* weights() const override
        { return layer_.front()->weights(); }
    virtual Float* weights() override
        { return layer_.front()->weights(); }
    virtual void setWeight(size_t input, size_t output, Float w) override
        { layer_.front()->setWeight(input, output, w); }

	// ------- layer interface ---------------

	/** return nr of layers */
    size_t numLayer() const;

    /** Adds a new layer, ownership IS TAKEN */
    void add(Layer<Float>* layer);

    /** Inserts a new layer before @p index, ownership IS TAKEN */
    void insert(size_t index, Layer<Float>* layer);

    /** To be called when a layer that has already been added
        has changed it's size. */
    void updateLayers() { resizeBuffers_(); }

    /** Returns the @p index'th layer */
    const Layer<Float>* layer(size_t index) const { return layer_[index]; }

    /** Returns the @p index'th layer */
    Layer<Float>* layer(size_t index) { return layer_[index]; }

	// ------- propagation -------------------

    virtual void fprop(const Float * input, Float * output) override;

    virtual void bprop(const Float * error, Float * error_output = 0,
                       Float global_learn_rate = 1) override;

	// ------- info --------------------------

    virtual const char * id() const override { return "StackSerial"; }
    virtual const char * name() const override { return "StackSerial"; }
    virtual size_t numParameters() const override;
    virtual void info(std::ostream &out = std::cout) const override;
    virtual void dump(std::ostream &out = std::cout) const override;

    virtual Float getWeightAverage() const override;

    // ------------- io ---------------

    virtual void serialize(std::ostream&) const override;
    virtual void deserialize(std::istream&) override;

protected:

	void resizeBuffers_();

	/** intermediate buffers */
	std::vector<std::vector<Float> > buffer_;

	/** inidividual layers */
	std::vector<Layer<Float>*> layer_;
};

#include "stack_serial_impl.inl"

} // namespace MNN

#endif // MNN_STACK_SERIAL_H_INCLUDED
