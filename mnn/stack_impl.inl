/** @file

    @brief

    <p>(c) 2016, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 1/19/2016</p>
*/


#define MNN_TEMPLATE template <typename Float>
#define MNN_STACK Stack<Float>


MNN_TEMPLATE
void MNN_STACK::clearLayers()
{
    // free layers
    for (auto l : layer_)
        l->releaseRef();
    layer_.clear();
}

MNN_TEMPLATE
void MNN_STACK::addLayer(Layer<Float> * layer, bool addRef)
{
    if (addRef)
        layer->addRef();
    layer_.push_back(layer);
    updateLayers();
}

MNN_TEMPLATE
void MNN_STACK::insertLayer(size_t index, Layer<Float> * layer, bool addRef)
{
    if (addRef)
        layer->addRef();

    if (index >= layer_.size())
        layer_.push_back(layer);
    else
        layer_.insert(layer_.begin() + index, layer);

    updateLayers();
}





// ----------- nn interface --------------


MNN_TEMPLATE
void MNN_STACK::resize(size_t numIn, size_t numOut)
{
    if (layer_.empty())
        return;

    if (layer_.size() == 1)
        layer_[0]->resize(numIn, numOut);
    else
    {
        layer_[0]->resize(numIn, layer_[0]->numOut());
        layer_[layer_.size()-1]->resize(layer_[layer_.size()-1]->numIn(), numOut);
    }

    updateLayers();
}

MNN_TEMPLATE
void MNN_STACK::grow(size_t nrIn, size_t nrOut, Float randomDev)
{
    if (layer_.empty())
        return;

    if (layer_.size() == 1)
        layer_[0]->grow(nrIn, nrOut, randomDev);
    else
    {
        if (nrIn > layer_.front()->numIn())
            layer_.front()->grow(nrIn, layer_.front()->numOut(), randomDev);
        if (nrOut > layer_.back()->numOut())
            layer_.back()->grow(layer_.back()->numIn(), nrOut, randomDev);
    }

    updateLayers();
}

MNN_TEMPLATE
size_t MNN_STACK::numIn() const
{
    return (layer_.empty())? 0 : layer_[0]->numIn();
}

MNN_TEMPLATE
size_t MNN_STACK::numOut() const
{
    return (layer_.empty())? 0 : layer_.back()->numOut();
}


MNN_TEMPLATE
void MNN_STACK::brainwash(Float amp)
{
    for (auto l : layer_)
        l->brainwash(amp);
}




MNN_TEMPLATE
void MNN_STACK::setDropOutMode(DropOutMode m)
{
    for (auto l : layer_)
        if (auto d = dynamic_cast<SetDropOutInterface<Float>*>(l))
            d->setDropOutMode(m);
}

MNN_TEMPLATE
void MNN_STACK::setDropOut(Float prob)
{
    for (auto l : layer_)
        if (auto d = dynamic_cast<SetDropOutInterface<Float>*>(l))
            d->setDropOut(prob);
}

MNN_TEMPLATE
void MNN_STACK::setMomentum(Float m)
{
    for (auto l : layer_)
        if (auto d = dynamic_cast<SetMomentumInterface<Float>*>(l))
            d->setMomentum(m);
}

MNN_TEMPLATE
void MNN_STACK::setSoftmax(bool e)
{
    for (auto l : layer_)
        if (auto d = dynamic_cast<SetSoftmaxInterface*>(l))
            d->setSoftmax(e);
}


// ----------------- info ------------------

MNN_TEMPLATE
Float MNN_STACK::getWeightAverage() const
{
    Float a = 0.;
    for (auto l : layer_)
        a += l->getWeightAverage();
    if (!layer_.empty())
        a /= layer_.size();
    return a;
}

MNN_TEMPLATE
size_t MNN_STACK::numParameters() const
{
    size_t num = 0;
    for (auto l : layer_)
        num += l->numParameters();
    return num;
}


#undef MNN_TEMPLATE
#undef MNN_STACK
