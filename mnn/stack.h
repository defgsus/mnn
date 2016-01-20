/** @file

    @brief

    <p>(c) 2016, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 1/19/2016</p>
*/

#ifndef MNNSRC_STACK_H
#define MNNSRC_STACK_H

#include <vector>

#include "layer.h"
#include "interface.h"

namespace MNN {

/** Base class of layer collections */
template <typename Float>
class Stack
        : public Layer<Float>
        , public SetMomentumInterface<Float>
        , public SetDropOutInterface<Float>
        , public SetLearnRateInterface<Float>
        , public SetLearnRateBiasInterface<Float>
        , public SetSoftmaxInterface
{
public:

    // ------- layer interface ---------------

    /** To be called when a layer that has already been added
        has changed it's size.
        This is also called by the default implementations of
        addLayer() and insertLayer(). */
    virtual void updateLayers() = 0;

    /** Removes all layers and references */
    virtual void clearLayers();

    /** Returns number of layers */
    virtual size_t numLayers() const { return layer_.size(); }

    /** Adds a new layer, and adds a reference to the layer. */
    virtual void addLayer(Layer<Float>* layer, bool addReference = false);

    /** Inserts a new layer before @p index, and adds a reference to the layer. */
    virtual void insertLayer(size_t index, Layer<Float>* layer, bool addReference = false);

    /** Returns the @p index'th layer */
    virtual const Layer<Float>* layer(size_t index) const { return layer_[index]; }

    /** Returns the @p index'th layer */
    virtual Layer<Float>* layer(size_t index) { return layer_[index]; }

    // ------------ nn interface -------------

    /** Calls the appropriate resize() function for first and last layer */
    virtual void resize(size_t numIn, size_t numOut) override;
    /** Grows the first and the last layer */
    virtual void grow(size_t nrIn, size_t nrOut, Float randomDev) override;
    /** Brainwashes ALL layers */
    virtual void brainwash(Float variance = 1.) override;

    /** Number of inputs of first layer */
    virtual size_t numIn() const override;
    /** Number of outputs of last layer */
    virtual size_t numOut() const override;

    virtual const Float* inputs() const override
        { return layer_.empty() ? 0 : layer_.front()->inputs(); }
    virtual const Float* outputs() const override
        { return layer_.empty() ? 0 : layer_.back()->outputs(); }
    virtual const Float* weights() const override
        { return layer_.empty() ? 0 : layer_.front()->weights(); }
    virtual Float* weights() override
        { return layer_.empty() ? 0 : layer_.front()->weights(); }
    virtual void setWeight(size_t input, size_t output, Float w) override
        { if (!layer_.empty()) layer_.front()->setWeight(input, output, w); }

    // --------- LearnRateInterface ----------

    /** Sets learnrate for ALL layers */
    virtual void setLearnRate(Float lr) override;

    // --------- LearnRateBiasInterface ------

    /** Sets bias learnrate for ALL layers */
    virtual void setLearnRateBias(Float lr) override;

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


    // ------- info --------------------------

    /** Sum of parameters in all layers */
    virtual size_t numParameters() const override;

    /** Average of all weights in all layers */
    virtual Float getWeightAverage() const override;

protected:

    std::vector<Layer<Float>*> layer_;
};


#include "stack_impl.inl"

} // namespace MNN

#endif // MNNSRC_STACK_H

