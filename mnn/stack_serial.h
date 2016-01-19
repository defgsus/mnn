/**	@file

	@brief Serial Stack header

	@author def.gsus-
	@version 2012/10/19 started
*/
#ifndef MNNSRC_STACK_SERIAL_H_INCLUDED
#define MNNSRC_STACK_SERIAL_H_INCLUDED

#include <cmath>
#include <vector>
#include <iostream>

#include "layer.h"
#include "interface.h"

namespace MNN {

template <typename Float>
class StackSerial
        : public Layer<Float>
        , public SetMomentumInterface<Float>
        , public SetDropOutInterface<Float>
        , public SetSoftmaxInterface
{
	public:

	StackSerial();

	virtual ~StackSerial();

    // ----------- copying -------------------

    virtual StackSerial<Float> * cloneClass() const override
        { return new StackSerial<Float>(); }

    virtual StackSerial<Float>& operator = (const Layer<Float>&) override;

    // --------- MomentumInterface -----------

    /** Sets momentum for ALL layers */
    virtual void setMomentum(Float m) override;

    // --------- DropOutInterface ------------

    /** Sets dropout mode for ALL layers */
    virtual void setDropOutMode(DropOutMode m) override;

    /** Sets dropout probability for ALL layers */
    virtual void setDropOut(Float probability) override;

    // -------- SetSoftmaxInterface ----------

    /** Sets the softmax mode for ALL layers */
    virtual void setSoftmax(bool enable) override;

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

    /** Destroys all layers */
    void clearLayers();

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

    static const char* static_id() { return "stack_serial"; }
    virtual const char * id() const override { return static_id(); }
    virtual const char * name() const override { return "StackSerial"; }
    virtual size_t numParameters() const override;
    virtual void info(std::ostream &out = std::cout,
                      const std::string& postFix = "") const override;
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

#endif // MNNSRC_STACK_SERIAL_H_INCLUDED
