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
    virtual void brainwash() override;

    virtual size_t numIn() const override;
    virtual size_t numOut() const override;
    virtual const Float* input() const override
        { return layer_.empty() ? 0 : layer_.front()->input(); }
    virtual const Float* output() const override
        { return layer_.empty() ? 0 : layer_.back()->output(); }

    using Layer<Float>::input;
    using Layer<Float>::output;

	// ------- layer interface ---------------

	/** return nr of layers */
    virtual size_t numLayer() const;

    /** Adds a new layer, ownership is taken */
	virtual void add(Layer<Float> * layer);

	// ------- propagation -------------------

    virtual void fprop(const Float * input, Float * output) override;

    virtual void bprop(const Float * error, Float * error_output = 0,
                       Float global_learn_rate = 1) override;

	// ------- info --------------------------

    virtual const char * id() const override { return "StackSerial"; }
    virtual const char * name() const override { return "StackSerial"; }
    virtual void info(std::ostream &out = std::cout) const override;
    virtual void dump(std::ostream &out = std::cout) const override;

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
